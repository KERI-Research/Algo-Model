from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import main as api_main  # noqa: E402


class BiomarkerRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api_main.app)

    def test_biomarker_route_requests_required_fields(self) -> None:
        response = self.client.post(
            "/api/v1/biomarker-discovery",
            json={
                "dataset": "nhanes_merged.csv",
                "patient_record": {
                    "Diabetes": 1,
                    "DEMO_RIDAGEYR": 62,
                },
                "top_k": 5,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "needs_required_fields")
        self.assertIn("BMX_BMXBMI", assessment["missing_required_fields"])
        self.assertGreaterEqual(len(assessment["follow_up_questions"]), 1)

    def test_biomarker_route_returns_scored_payload(self) -> None:
        response = self.client.post(
            "/api/v1/biomarker-discovery",
            json={
                "dataset": "nhanes_merged.csv",
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
        self.assertIn(payload["model"], payload["benchmarks"])
        self.assertEqual(len(payload["biomarker_ranking"]), 5)
        assessment = payload["patient_assessment"]
        self.assertEqual(assessment["status"], "scored")
        self.assertGreaterEqual(assessment["cancer_risk_probability"], 0.0)
        self.assertLessEqual(assessment["cancer_risk_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()