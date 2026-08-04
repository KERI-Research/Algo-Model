"""Model probe outputs, research surfaces and claim safety."""

from __future__ import annotations

import json

from conftest import TEST_ACCESS_KEY  # noqa: F401  (fixture dependency)

from server import config, model, reports

SAMPLE_RECORD = {
    "DEMO_RIDAGEYR": 61,
    "DEMO_RIAGENDR": 1,
    "BMX_BMXBMI": 33.4,
    "BMX_BMXWAIST": 112.0,
    "GHB_LBXGH": 6.4,
    "GLU_LBXGLU": 128,
    "INS_LBXIN": 21.5,
    "TRIGLY_LBXTR": 210,
    "HDL_LBDHDD": 38,
    "TCHOL_LBXTC": 214,
    "HSCRP_LBXHSCRP": 4.1,
    "smoking_status": 1,
    "homa_ir": 6.8,
}


def test_no_torch_import_in_deployment():
    import sys

    model.score_single_record(SAMPLE_RECORD)
    assert "torch" not in sys.modules


def test_score_single_record_outputs():
    result = model.score_single_record(SAMPLE_RECORD)
    score = result["score"]
    assert score["metabolic_deviation_score"] >= 0
    assert 0 <= score["reference_percentile"] <= 100
    assert len(score["latent_representation"]) == 16
    assert score["is_future_risk_probability"] is False
    assert result["is_disease_classification"] is False
    assert result["features_used"]
    assert result["evidence_boundaries"]
    assert "deviation" in result["field_meanings"]["metabolic_deviation_score"].lower()


def test_score_single_record_ignores_prohibited_fields():
    payload = dict(SAMPLE_RECORD)
    payload.update({"Cancer": 1, "tcga_stage_ordinal": 3, "PancreaticCancer": 1})
    result = model.score_single_record(payload)
    assert "Cancer" not in result["features_used"]
    assert not any(name.startswith("tcga_") for name in result["features_used"])


def test_score_single_record_requires_allowlisted_values():
    try:
        model.score_single_record({"unknown_column": 5})
    except ValueError as error:
        assert "allowlisted" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_probe_endpoint_requires_explicit_scoring(authed_client):
    response = authed_client.post(
        "/api/v1/probe/score",
        json={"patient_record": SAMPLE_RECORD, "confirm_explicit_scoring": False},
    )
    assert response.status_code == 400
    ok = authed_client.post(
        "/api/v1/probe/score",
        json={"patient_record": SAMPLE_RECORD, "confirm_explicit_scoring": True},
    )
    assert ok.status_code == 200


def test_model_summary_states_prohibited_outputs(authed_client):
    body = authed_client.get("/api/v1/model").json()
    assert body["inference_backend"] == "numpy"
    assert body["model_version"] == config.MODEL_VERSION
    assert body["architecture"]["latent_dimension"] == 16
    assert any("cancer type" in item for item in body["prohibited_outputs"])
    assert "non_diagnostic_warning" in body


def test_overview_reports_limitations_and_cards(authed_client):
    body = authed_client.get("/api/v1/overview").json()
    ids = {card["id"] for card in body["status_cards"]}
    assert ids == {"inference", "reliability", "clustering", "horizon"}
    assert {item["id"] for item in body["data_limitations"]} >= {
        "cross_sectional",
        "prevalent_labels",
        "survey_cycle",
    }
    assert body["posture"]["headline"].lower().startswith("non-diagnostic")


def test_reliability_report_surface(authed_client):
    body = authed_client.get("/api/v1/reliability").json()
    assert body["status"] == "ok"
    assert set(body["tier_definitions"]) == {
        "usable_now",
        "qualified_use",
        "unavailable",
        "prohibited",
    }
    assert body["tiers"]["usable_now"]
    assert body["fail_closed_controls"]
    assert body["explanation_class"] == "data_observation"


def test_clustering_abstains_with_survey_cycle_explanation(authed_client):
    body = authed_client.get("/api/v1/clusters").json()
    assert body["status"] == "no_stable_clusters"
    assert body["is_disease_classification"] is False
    assert "survey_cycle" in json.dumps(body["abstain"]["gate_failure_summary"])
    assert "cycle" in body["abstain"]["survey_cycle_explanation"].lower()
    assert "clusters" not in body


def test_clustering_variant_validation(authed_client):
    assert authed_client.get("/api/v1/clusters?variant=all_adults").status_code == 200
    missing = authed_client.get("/api/v1/clusters?variant=made_up")
    assert missing.status_code == 404
    assert missing.json()["error"]["available_variants"] == ["complete_cases", "all_adults"]


def test_evidence_catalogue_entries_are_source_linked(authed_client):
    body = authed_client.get("/api/v1/evidence").json()
    assert body["summary"]["entry_count"] > 0
    assert body["clinician_ready_entries"]
    for entry in body["clinician_ready_entries"]:
        assert entry.get("primary_source_url", "").startswith("http") or entry.get("doi")
        assert entry.get("evidence_grade")
    assert any("PRoBE" in ref["title"] for ref in body["method_references"])
    assert any("TRIPOD" in ref["title"] for ref in body["method_references"])
    assert body["supported_claims"] and body["prohibited_claims"]
    assert "catalogue_path" not in body["summary"]


def test_results_csv_headers_and_rows():
    rows = [
        {
            "row_number": 1,
            "row_index": 4,
            "metabolic_deviation_score": 1.5,
            "reference_percentile": 88.2,
            "top_deviation_features": [{"feature": "homa_ir", "reconstruction_error": 0.4}],
        }
    ]
    body = model.results_csv(rows)
    header, first = body.strip().splitlines()
    assert header.split(",")[:4] == [
        "row_number",
        "source_row_index",
        "metabolic_deviation_score",
        "reference_percentile",
    ]
    assert first.startswith("1,4,1.5,88.2,homa_ir,,")


def test_reports_never_leak_server_paths(authed_client):
    for path in ("/api/v1/reliability", "/api/v1/evidence", "/api/v1/clusters", "/api/v1/model"):
        body = authed_client.get(path).text
        assert "/home/" not in body
        assert "/Volumes/" not in body
