from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from biomarker import (  # noqa: E402
    ALTERNATE_MODEL_NAME,
    MODEL_NAME,
    XGBClassifier,
    _normalize_patient_record,
    execute_biomarker_discovery,
    train_biomarker_model,
)

PROJECT_ROOT = API_DIR.parent
# Corrected dataset only: nhanes_merged.csv is invalidated (MCQ230 39 vs 29).
DATASET_PATH = PROJECT_ROOT / "data" / "nhanes_merged_v2.csv"


def artifacts_writable() -> bool:
    root = API_DIR / "model_artifacts"
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root):
            return True
    except OSError:
        return False


class InvalidatedInputTests(unittest.TestCase):
    def test_invalidated_dataset_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            train_biomarker_model(str(PROJECT_ROOT / "data" / "nhanes_merged.csv"))

    def test_invalidated_target_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            train_biomarker_model(str(DATASET_PATH), target="PancreaticCancer")

    def test_normalize_patient_record_handles_optional_field_not_in_features(self) -> None:
        artifact = {
            "features": ["Diabetes"],
            "required_fields": ["Diabetes"],
            "optional_high_impact_fields": ["CPEP_LBXCPSI"],
            "medians": {"Diabetes": 0.0},
        }

        patient_features, missingness = _normalize_patient_record(
            {"Diabetes": 1},
            artifact,
        )

        self.assertEqual(list(patient_features.columns), ["Diabetes"])
        self.assertEqual(missingness["missing_required_fields"], [])
        self.assertIn("CPEP_LBXCPSI", missingness["missing_optional_fields"])


@unittest.skipUnless(artifacts_writable(), "artifact directory is not writable")
@unittest.skipUnless(DATASET_PATH.exists(), "corrected dataset not present")
class BiomarkerFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_path = DATASET_PATH
        cls.artifact = train_biomarker_model(str(cls.dataset_path), force=False)

    def test_artifact_contains_lab_features_and_benchmarks(self) -> None:
        self.assertIn("GHB_LBXGH", self.artifact["features"])
        self.assertIn("GLU_LBXGLU", self.artifact["features"])
        self.assertIn("INS_LBXIN", self.artifact["features"])
        # Canonical naming contract: benchmark keys are the MetaboGuard model names
        # exported by biomarker.py. The legacy key
        # "hist_gradient_boosting_biomarker_v1" predates the MetaboGuard rename and is
        # not produced or consumed anywhere, so it is not kept as an alias.
        self.assertIn(MODEL_NAME, self.artifact["benchmarks"])
        if XGBClassifier is not None:
            self.assertIn(ALTERNATE_MODEL_NAME, self.artifact["benchmarks"])

    def test_model_names_follow_the_metaboguard_contract(self) -> None:
        self.assertEqual(MODEL_NAME, "metaboguard_hist_gradient_boosting_v1")
        self.assertEqual(ALTERNATE_MODEL_NAME, "metaboguard_xgboost_v1")
        for name in self.artifact["benchmarks"]:
            self.assertTrue(
                name.startswith("metaboguard_"),
                f"benchmark key {name!r} breaks the metaboguard_* naming contract",
            )
        # The selected model must be one of the benchmarked candidates.
        self.assertIn(self.artifact["model_name"], self.artifact["benchmarks"])

    def test_target_is_never_used_as_its_own_feature(self) -> None:
        self.assertNotIn(self.artifact["target"], self.artifact["features"])

    def test_artifact_is_tied_to_a_dataset_fingerprint(self) -> None:
        self.assertEqual(len(self.artifact["dataset_signature"]), 64)

    def test_incomplete_record_requests_required_fields(self) -> None:
        payload = execute_biomarker_discovery(
            str(self.dataset_path),
            patient_record={"Diabetes": 1, "DEMO_RIDAGEYR": 62},
            top_k=5,
        )

        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "needs_required_fields")
        self.assertIn("BMX_BMXBMI", assessment["missing_required_fields"])
        self.assertGreaterEqual(len(assessment["follow_up_questions"]), 1)
        self.assertIn("data_readiness", assessment)
        self.assertEqual(
            assessment["data_readiness"]["dataset_capability_state"],
            "Cross-sectional only",
        )
        self.assertEqual(assessment["safety_contract"]["future_risk"], "disabled")

    def test_complete_record_returns_cross_sectional_association(self) -> None:
        payload = execute_biomarker_discovery(
            str(self.dataset_path),
            patient_record={
                "Diabetes": 1,
                "DEMO_RIDAGEYR": 62,
                "DEMO_RIAGENDR": 2,
                "BMX_BMXBMI": 31.4,
                "BMX_BMXWAIST": 101.2,
                "DIQ_DID040": 56,
                "GHB_LBXGH": 6.8,
                "GLU_LBXGLU": 134,
                "INS_LBXIN": 18.5,
            },
            top_k=5,
        )

        self.assertEqual(payload["output_type"], "cross_sectional_association")
        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "scored")
        self.assertEqual(assessment["output_type"], "cross_sectional_association")
        self.assertFalse(assessment["is_future_risk_probability"])
        self.assertGreaterEqual(assessment["cross_sectional_association_probability"], 0.0)
        self.assertLessEqual(assessment["cross_sectional_association_probability"], 1.0)
        self.assertIn("not a probability of developing disease", assessment["explanation"])
        self.assertIn("current_profile_assessment", assessment)
        self.assertIn("standout_factors", assessment)
        self.assertIn("association_scope", assessment)
        self.assertIn("data_readiness", assessment)
        self.assertEqual(
            payload["dataset_capability_state"],
            "Cross-sectional only",
        )
        self.assertEqual(assessment["safety_contract"]["future_risk"], "disabled")
        self.assertIn(payload["model"], payload["benchmarks"])
        self.assertEqual(len(payload["biomarker_ranking"]), 5)


if __name__ == "__main__":
    unittest.main()