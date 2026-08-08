"""Simulation-only future-risk deployment contract."""

from __future__ import annotations

import json

from conftest import TEST_ACCESS_KEY
from server import future_risk


def synthetic_visits() -> list[dict[str, float | int]]:
    return [
        {
            "visit_index": 0,
            "days_before_index": 820,
            "years_before_index": 2.25,
            "DEMO_RIDAGEYR": 52.8,
            "BMX_BMXBMI": 29.4,
            "BMX_BMXWAIST": 101.2,
            "BMX_BMXWT": 88.1,
            "GHB_LBXGH": 5.8,
            "GLU_LBXGLU": 103,
            "INS_LBXIN": 12.4,
            "TCHOL_LBXTC": 204,
            "HDL_LBDHDD": 44,
            "TRIGLY_LBXTR": 166,
            "BPX_SYSTOLIC": 128,
        },
        {
            "visit_index": 1,
            "days_before_index": 390,
            "years_before_index": 1.07,
            "DEMO_RIDAGEYR": 54.0,
            "BMX_BMXBMI": 30.1,
            "BMX_BMXWAIST": 103.8,
            "BMX_BMXWT": 90.0,
            "GHB_LBXGH": 6.0,
            "GLU_LBXGLU": 109,
            "INS_LBXIN": 14.1,
            "TCHOL_LBXTC": 210,
            "HDL_LBDHDD": 42,
            "TRIGLY_LBXTR": 184,
            "BPX_SYSTOLIC": 132,
        },
        {
            "visit_index": 2,
            "days_before_index": 0,
            "years_before_index": 0,
            "DEMO_RIDAGEYR": 55.1,
            "BMX_BMXBMI": 31.0,
            "BMX_BMXWAIST": 106.4,
            "BMX_BMXWT": 92.2,
            "GHB_LBXGH": 6.2,
            "GLU_LBXGLU": 116,
            "INS_LBXIN": 16.8,
            "TCHOL_LBXTC": 218,
            "HDL_LBDHDD": 40,
            "TRIGLY_LBXTR": 205,
            "BPX_SYSTOLIC": 136,
        },
    ]


def test_portable_bundle_is_small_and_contains_no_authoritative_model():
    files = list(future_risk.PORTABLE_DIR.iterdir())
    assert future_risk.artifact_available() is True
    assert sum(path.stat().st_size for path in files) < 1_000_000
    assert not list(future_risk.PORTABLE_DIR.glob("*.joblib"))
    assert not list(future_risk.PORTABLE_DIR.glob("*.pt"))
    assert not list(future_risk.PORTABLE_DIR.glob("*.pth"))
    assert not (future_risk.PORTABLE_DIR / "future_risk_models.joblib").exists()

    manifest = json.loads(future_risk.PORTABLE_JSON.read_text())
    assert manifest["simulation_only"] is True
    assert manifest["clinical_use"] == "prohibited"
    assert set(manifest["models"]) == set(manifest["supported"])


def test_capability_is_protected_and_reports_measured_parity(client):
    assert client.get("/api/v1/simulation/capability").status_code == 401
    login = client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    assert login.status_code == 200

    response = client.get("/api/v1/simulation/capability")
    assert response.status_code == 200
    body = response.json()
    assert body["simulation_only"] is True
    assert body["clinical_use"] == "prohibited"
    assert body["portable_artifact_available"] is True
    assert body["minimum_visits"] == 2
    assert body["parity"]["verdict"] == "parity"
    assert body["parity"]["max_abs_difference"] <= 1e-6
    assert set(body["supported_horizons"]) == {
        "type2_diabetes:3y",
        "type2_diabetes:5y",
        "pan_cancer:5y",
    }
    assert set(body["abstained_horizons"]) == {
        "type2_diabetes:1y",
        "pan_cancer:1y",
        "pan_cancer:3y",
    }


def test_simulation_scores_supported_horizons_and_abstains_elsewhere(authed_client):
    response = authed_client.post(
        "/api/v1/simulation/score",
        json={
            "visits": synthetic_visits(),
            "simulation_mode": True,
            "seed": 17,
            "archetype": "metabolic_deviation",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["simulation_only"] is True
    assert body["clinical_use"] == "prohibited"
    assert body["inference_backend"] == "numpy_portable"
    assert body["history"]["visits"] == 3
    assert body["persistence"].startswith("none:")

    expected = {
        "type2_diabetes": {"1y": "abstained", "3y": "simulated_estimate", "5y": "simulated_estimate"},
        "pan_cancer": {"1y": "abstained", "3y": "abstained", "5y": "simulated_estimate"},
    }
    for outcome, horizons in expected.items():
        for horizon, status in horizons.items():
            result = body["outcomes"][outcome]["horizons"][horizon]
            assert result["status"] == status
            if status == "simulated_estimate":
                model = result["models"][result["selected_model"]]
                assert 0 <= model["raw_cumulative_incidence"] <= 1
                assert 0 <= model["calibrated_cumulative_incidence"] <= 1
            else:
                assert result["selected_model"] is None
                assert result["models"] == {}


def test_simulation_refuses_cross_sectional_or_unmarked_input(authed_client):
    unmarked = authed_client.post(
        "/api/v1/simulation/score",
        json={"visits": synthetic_visits(), "simulation_mode": False},
    )
    assert unmarked.status_code == 422
    assert "simulation_mode must be true" in unmarked.json()["error"]["message"]

    cross_sectional = authed_client.post(
        "/api/v1/simulation/score",
        json={"visits": [synthetic_visits()[0]], "simulation_mode": True},
    )
    assert cross_sectional.status_code == 422
    assert "single visit is cross-sectional" in cross_sectional.json()["error"]["message"]


def test_simulation_refuses_identifier_and_free_text_fields(authed_client):
    visits = synthetic_visits()
    visits[0]["full_name"] = "Not Allowed"
    visits[0]["visit_label"] = "free text is not accepted"
    response = authed_client.post(
        "/api/v1/simulation/score",
        json={"visits": visits, "simulation_mode": True},
    )
    assert response.status_code == 422
    assert response.json()["error"]["rejected_fields"] == ["full_name", "visit_label"]

    arbitrary_archetype = authed_client.post(
        "/api/v1/simulation/score",
        json={
            "visits": synthetic_visits(),
            "simulation_mode": True,
            "archetype": "free-form patient note",
        },
    )
    assert arbitrary_archetype.status_code == 422


def test_clinical_future_risk_route_always_refuses(authed_client):
    response = authed_client.post("/api/v1/future-risk/score", json={})
    assert response.status_code == 409
    body = response.json()["error"]
    assert body["capability_state"] == "simulation_only_longitudinal"
    assert body["use_instead"] == "/api/v1/simulation/score"
    assert "patient-facing risk estimate" in body["message"]