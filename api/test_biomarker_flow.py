from __future__ import annotations

import unittest
from pathlib import Path

from api.biomarker import execute_biomarker_discovery, train_biomarker_model


class BiomarkerFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parent.parent
        cls.dataset_path = project_root / "api" / "nhanes_data" / "nhanes_merged.csv"
        cls.artifact = train_biomarker_model(str(cls.dataset_path), force=False)

    def test_artifact_contains_lab_features_and_benchmarks(self) -> None:
        self.assertIn("GHB_LBXGH", self.artifact["features"])
        self.assertIn("GLU_LBXGLU", self.artifact["features"])
        self.assertIn("INS_LBXIN", self.artifact["features"])
        self.assertIn("hist_gradient_boosting_biomarker_v1", self.artifact["benchmarks"])

    def test_incomplete_record_requests_required_fields(self) -> None:
        payload = execute_biomarker_discovery(
            str(self.dataset_path),
            patient_record={
                "Diabetes": 1,
                "DEMO_RIDAGEYR": 62,
            },
            top_k=5,
        )

        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "needs_required_fields")
        self.assertIn("BMX_BMXBMI", assessment["missing_required_fields"])
        self.assertGreaterEqual(len(assessment["follow_up_questions"]), 1)

    def test_complete_record_returns_score_and_benchmark_payload(self) -> None:
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

        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "scored")
        self.assertGreaterEqual(assessment["cancer_risk_probability"], 0.0)
        self.assertLessEqual(assessment["cancer_risk_probability"], 1.0)
        self.assertIn(payload["model"], payload["benchmarks"])
        self.assertEqual(len(payload["biomarker_ranking"]), 5)


if __name__ == "__main__":
    unittest.main()