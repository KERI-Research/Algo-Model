from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient


API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import main as api_main  # noqa: E402

# nhanes_merged.csv and nhanes_multicycle.csv are invalidated (MCQ230 code 39 was
# treated as pancreas), so route tests use the corrected v2 file.
DATASET = "nhanes_merged_v2.csv"


def artifacts_writable() -> bool:
    """Supervised routes train on first call, which needs a writable artifact dir."""
    root = API_DIR / "model_artifacts"
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root):
            return True
    except OSError:
        return False


class DatasetGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api_main.app)

    def test_invalidated_dataset_is_not_listed(self) -> None:
        names = {item["name"] for item in self.client.get("/api/v1/datasets").json()["datasets"]}
        self.assertNotIn("nhanes_merged.csv", names)
        self.assertNotIn("nhanes_multicycle.csv", names)
        self.assertIn("nhanes_multicycle_v2.csv", names)

    def test_invalidated_dataset_is_refused(self) -> None:
        response = self.client.post(
            "/api/v1/dataset-preview", json={"dataset": "nhanes_multicycle.csv"}
        )
        self.assertEqual(response.status_code, 404)

    def test_capabilities_route_is_truthful(self) -> None:
        payload = self.client.post("/api/v1/prevention-capabilities", json={}).json()
        self.assertFalse(payload["longitudinal_heads_enabled"])
        self.assertFalse(payload["capabilities"]["supports_future_development_prediction"])
        self.assertEqual(
            payload["capabilities"]["supported_output"],
            "cross_sectional_representation_and_deviation_only",
        )
        self.assertIn("non-diagnostic", payload["non_diagnostic_warning"].lower())
        self.assertIn("PancreaticCancer", payload["invalidated_supervised_targets"])
        for horizon in ("365d", "1095d", "1825d"):
            self.assertFalse(payload["future_horizon_gates"]["per_horizon"][horizon]["eligible"])

    def test_future_risk_route_is_fail_closed(self) -> None:
        response = self.client.post("/api/v1/prevention-future-risk", json={})
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["intended_horizons_days"], [365, 1095, 1825])
        self.assertIn("disabled", detail["message"])

    def test_data_integrity_route_reports_corrected_coding(self) -> None:
        payload = self.client.post(
            "/api/v1/data-integrity", json={"dataset": "nhanes_multicycle_v2.csv"}
        ).json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["cancer_coding"]["matches_code_29"])
        self.assertEqual(payload["cancer_coding"]["rows_with_code_29_pancreas"], 19)


class PreventionScoreRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api_main.app)

    def test_score_route_returns_deviation_not_probability(self) -> None:
        response = self.client.post(
            "/api/v1/prevention-score",
            json={
                "patient_record": {
                    "DEMO_RIDAGEYR": 61,
                    "BMX_BMXBMI": 33.1,
                    "GHB_LBXGH": 7.1,
                    "GLU_LBXGLU": 141,
                }
            },
        )
        if response.status_code == 409:
            self.skipTest("No trained SSL artifact available in this checkout.")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_type"], "metabolic_deviation_and_representation")
        self.assertFalse(payload["is_future_risk_probability"])
        self.assertIn("non-diagnostic", payload["clinical_warning"].lower())
        score = payload["score"]
        self.assertIn("metabolic_deviation_score", score)
        self.assertIn("reference_percentile", score)
        self.assertIn("latent_representation", score)
        self.assertNotIn("risk_probability", score)

    def test_missing_artifact_returns_conflict_not_a_fallback_score(self) -> None:
        response = self.client.post(
            "/api/v1/prevention-score",
            json={"patient_record": {"DEMO_RIDAGEYR": 61}, "artifact": "does_not_exist"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("not trained", response.json()["detail"])


@unittest.skipUnless(artifacts_writable(), "artifact directory is not writable")
class BiomarkerRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api_main.app)

    def test_biomarker_route_requests_required_fields(self) -> None:
        response = self.client.post(
            "/api/v1/biomarker-discovery",
            json={
                "dataset": DATASET,
                "patient_record": {"Diabetes": 1, "DEMO_RIDAGEYR": 62},
                "top_k": 5,
            },
        )

        self.assertEqual(response.status_code, 200)
        assessment = response.json()["patient_assessment"]
        self.assertEqual(assessment["status"], "needs_required_fields")
        self.assertIn("BMX_BMXBMI", assessment["missing_required_fields"])
        self.assertGreaterEqual(len(assessment["follow_up_questions"]), 1)

    def test_biomarker_route_returns_cross_sectional_association(self) -> None:
        response = self.client.post(
            "/api/v1/biomarker-discovery",
            json={
                "dataset": DATASET,
                "patient_record": {
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
                "top_k": 5,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_type"], "cross_sectional_association")
        self.assertIn("not future-risk performance", payload["non_diagnostic_warning"])
        self.assertIn(payload["model"], payload["benchmarks"])
        self.assertEqual(len(payload["biomarker_ranking"]), 5)
        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "scored")
        self.assertFalse(assessment["is_future_risk_probability"])
        self.assertEqual(assessment["output_type"], "cross_sectional_association")
        self.assertGreaterEqual(assessment["cross_sectional_association_probability"], 0.0)
        self.assertLessEqual(assessment["cross_sectional_association_probability"], 1.0)
        self.assertIn("ALREADY has", assessment["explanation"])

    def test_pancreatic_target_is_refused_by_the_route(self) -> None:
        response = self.client.post(
            "/api/v1/biomarker-discovery",
            json={"dataset": DATASET, "target": "PancreaticCancer", "top_k": 5},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("disabled", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()