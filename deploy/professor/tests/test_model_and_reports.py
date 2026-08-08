"""Model probe outputs, research surfaces and claim safety."""

from __future__ import annotations

import json

import pytest

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


@pytest.mark.parametrize(
    ("percentile", "expected_band"),
    [
        (0, "within_reference_range"),
        (89.999, "within_reference_range"),
        (90, "mild_deviation"),
        (94.999, "mild_deviation"),
        (95, "elevated_deviation"),
        (98.999, "elevated_deviation"),
        (99, "high_deviation"),
        (100, "high_deviation"),
    ],
)
def test_deviation_band_thresholds(percentile, expected_band):
    assert model._deviation_band_from_percentile(percentile)["key"] == expected_band


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
    assert result["dataset_capability_state"] == "Cross-sectional only"
    assessment = result["patient_assessment"]
    assert "current_profile_assessment" in assessment
    assert "standout_factors" in assessment
    assert "data_readiness" in assessment
    assert "research_association" in assessment
    assert assessment["safety_contract"]["future_risk"] == "disabled"
    current_profile = assessment["current_profile_assessment"]
    assert current_profile["note"] == model._deviation_band_from_percentile(
        score["reference_percentile"]
    )["interpretation"]
    interpretation = current_profile["deviation_interpretation"]
    assert interpretation["health_direction"] == "not_directional"
    assert "better or worse" in interpretation["health_direction_note"].lower()
    assert interpretation["range_review"]["status"] == "no_broad_range_flags"
    assert interpretation["range_review"]["flagged_values"] == []
    assert "combination" in interpretation["pattern_meaning"].lower()

    research = assessment["research_association"]
    assert [item["id"] for item in research["cancer_outcomes"]] == [
        "pan_cancer",
        "pancreatic_cancer",
        "other_site_specific_cancers",
    ]
    assert all(item["probability"] is None for item in research["cancer_outcomes"])
    assert research["cancer_outcomes"][0]["status"] == "simulation_only"
    assert all(
        item["status"] == "not_estimable"
        for item in research["cancer_outcomes"][1:]
    )
    assert "General cancers" not in json.dumps(research)
    assert [pathway["id"] for pathway in research["pathways"]] == [
        "diabetes_related_cancer",
        "lifestyle_related_cancer",
        "cancer_related_diabetes",
        "lifestyle_related_diabetes",
    ]
    assert all(pathway["status"] == "not_estimable" for pathway in research["pathways"])
    assert all(pathway["probability"] is None for pathway in research["pathways"])

    observed_ranked_features = [
        entry["feature"]
        for entry in assessment["standout_factors"]["top_deviation_features"]
    ]
    assert set(observed_ranked_features) <= set(result["features_used"])
    assert len(assessment["data_readiness"]["missing_fields"]) == len(
        result["features_missing"]
    )
    pathway_features = {
        definition["id"]: definition["features"]
        for definition in model.RESEARCH_PATHWAY_DEFINITIONS
    }
    for pathway in research["pathways"]:
        observed = [
            entry["feature"] for entry in pathway["observed_standout_features"]
        ]
        assert len(observed) <= 3
        assert set(observed) <= pathway_features[pathway["id"]]
        assert observed == [
            feature for feature in observed_ranked_features if feature in observed
        ]


def test_probe_flags_implausible_values_separately_from_deviation():
    payload = dict(SAMPLE_RECORD)
    payload["BMX_BMXBMI"] = 900
    result = model.score_single_record(payload)
    interpretation = result["patient_assessment"]["current_profile_assessment"][
        "deviation_interpretation"
    ]
    review = interpretation["range_review"]
    assert review["status"] == "review_flagged_values"
    assert review["flagged_values"] == [
        {
            "feature": "BMX_BMXBMI",
            "value": 900.0,
            "plausible_range": list(model.PLAUSIBLE_RANGES["BMX_BMXBMI"]),
        }
    ]
    assert "unit" in review["note"].lower()
    assert interpretation["health_direction"] == "not_directional"


def test_sparse_record_never_labels_imputed_features_as_observed():
    result = model.score_single_record({"DEMO_RIDAGEYR": 61})
    assessment = result["patient_assessment"]
    observed = assessment["standout_factors"]["top_deviation_features"]
    assert {entry["feature"] for entry in observed} <= {"DEMO_RIDAGEYR"}
    assert all(
        not pathway["observed_standout_features"]
        for pathway in assessment["research_association"]["pathways"]
    )


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


def test_probe_output_uses_non_future_risk_wording(authed_client):
    response = authed_client.post(
        "/api/v1/probe/score",
        json={"patient_record": SAMPLE_RECORD, "confirm_explicit_scoring": True},
    )
    assert response.status_code == 200
    body = response.json()
    text = json.dumps(body).lower()
    assert "risk of developing" not in text
    assert "prediction of future cancer" not in text
    assert "probability of getting" not in text
    assert "because of" not in text
    assert "occult cancer" not in text
    assert "later cancer" not in text
    assert "later diabetes" not in text
    assert "non-diagnostic" in body["non_diagnostic_warning"].lower()


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
