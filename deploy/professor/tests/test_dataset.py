"""Upload screening, limits, ephemerality, scoring and export."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from conftest import identifier_csv, leakage_csv, safe_csv, upload_files

from server import config, dataset


# --- identifier and leakage screening --------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        "name",
        "full_name",
        "first_name",
        "surname",
        "patient_name",
        "email",
        "Email Address",
        "phone",
        "mobile_number",
        "address",
        "address_line_1",
        "postcode",
        "zip_code",
        "nhs_number",
        "NHSNo",
        "ssn",
        "social_security_number",
        "mrn",
        "medical_record_number",
        "dob",
        "date_of_birth",
        "birthdate",
    ],
)
def test_direct_identifier_columns_are_detected(column):
    findings = dataset.screen_column_names([column, "DEMO_RIDAGEYR"])
    assert [item["column"] for item in findings] == [column]


@pytest.mark.parametrize(
    "column", ["age", "DEMO_RIDAGEYR", "row_id", "participant_id", "seqn", "index", "study_id"]
)
def test_age_and_anonymous_ids_are_allowed(column):
    assert dataset.screen_column_names([column]) == []


def test_identifier_values_are_detected_without_being_echoed():
    frame = pd.DataFrame({"contact": [f"user{i}@example.org" for i in range(10)]})
    findings = dataset.screen_column_values(frame)
    assert findings and findings[0]["identifier_type"] == "email address"
    assert "user0@example.org" not in str(findings)


def test_screen_dataset_rejects_identifier_file():
    frame = pd.read_csv(io.BytesIO(identifier_csv()))
    with pytest.raises(dataset.DatasetRejected) as error:
        dataset.screen_dataset(frame)
    assert "identifier" in str(error.value).lower()
    assert {item["column"] for item in error.value.detail["identifier_columns"]} >= {
        "full_name",
        "email",
    }


def test_leakage_columns_are_marked_prohibited_and_excluded():
    frame = pd.read_csv(io.BytesIO(leakage_csv()))
    screening = dataset.screen_dataset(frame)
    prohibited = {item["column"] for item in screening.prohibited_columns}
    assert {"Cancer", "PancreaticCancer", "tcga_stage_ordinal"} <= prohibited
    assert not prohibited & set(screening.mapped_features)


# --- limits ----------------------------------------------------------------


def test_non_csv_extension_is_rejected():
    with pytest.raises(dataset.DatasetRejected):
        dataset.read_csv_bytes(bytearray(b"a,b\n1,2\n"), "cohort.xlsx")


def test_empty_file_is_rejected():
    with pytest.raises(dataset.DatasetRejected):
        dataset.read_csv_bytes(bytearray(b""), "cohort.csv")


def test_oversize_byte_payload_is_rejected():
    payload = bytearray(b"x" * (config.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(dataset.DatasetRejected) as error:
        dataset.read_csv_bytes(payload, "cohort.csv")
    assert "upload limit" in str(error.value)


def test_row_cap_is_enforced_on_parse():
    header = "DEMO_RIDAGEYR\n"
    body = "".join("55\n" for _ in range(config.MAX_UPLOAD_ROWS + 5))
    with pytest.raises(dataset.DatasetRejected) as error:
        dataset.read_csv_bytes(bytearray((header + body).encode()), "cohort.csv")
    assert "row limit" in str(error.value)


def test_row_cap_boundary_is_accepted():
    header = "DEMO_RIDAGEYR\n"
    body = "".join("55\n" for _ in range(config.MAX_UPLOAD_ROWS))
    frame = dataset.read_csv_bytes(bytearray((header + body).encode()), "cohort.csv")
    assert len(frame) == config.MAX_UPLOAD_ROWS


def test_shred_buffer_zeroes_and_empties():
    buffer = bytearray(b"sensitive-bytes")
    dataset.shred_buffer(buffer)
    assert len(buffer) == 0
    dataset.shred_buffer(None)  # must not raise


# --- intake report ---------------------------------------------------------


def test_intake_report_shape():
    frame = pd.read_csv(io.BytesIO(safe_csv(50)))
    report = dataset.build_intake_report(frame, "cohort.csv")
    assert report["file"]["rows"] == 50
    assert "DEMO_RIDAGEYR" in report["schema"]["mapped_features"]
    assert "row_id" in report["schema"]["unmapped_columns"]
    assert report["rows"]["accepted"] == 50
    assert report["rows"]["rejected"] == 0
    assert report["model_ready"] is True
    assert report["dataset_capability"]["clustering_available_in_deployment"] is False
    assert report["dataset_capability"]["supports_future_development_prediction"] is False
    assert set(report["tier_definitions"]) == {
        "usable_now",
        "qualified_use",
        "unavailable",
        "prohibited",
    }
    assert report["feature_eligibility"]["CPEP_LBXCPSI"]["tier"] == "unavailable"


def test_range_violations_are_reported():
    payload = safe_csv(20).decode().splitlines()
    header = payload[0].split(",")
    index = header.index("GHB_LBXGH")
    row = payload[1].split(",")
    row[index] = "99"
    payload[1] = ",".join(row)
    frame = pd.read_csv(io.StringIO("\n".join(payload)))
    report = dataset.build_intake_report(frame, "cohort.csv")
    assert report["range_violations"]["GHB_LBXGH"]["values_outside_range"] == 1


def test_rows_without_enough_features_are_rejected():
    csv_text = "DEMO_RIDAGEYR,BMX_BMXBMI,GHB_LBXGH\n55,28,5.6\n,,\n60,,\n"
    frame = pd.read_csv(io.StringIO(csv_text))
    report = dataset.build_intake_report(frame, "cohort.csv")
    assert report["rows"]["accepted"] == 1
    assert report["rows"]["rejected"] == 2


# --- endpoint behaviour ----------------------------------------------------


def test_inspect_requires_the_deidentification_checkbox(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/inspect",
        files=upload_files(safe_csv(5)),
        data={"deidentified_confirmed": "false"},
    )
    assert response.status_code == 400
    assert "de-identified" in response.json()["error"]


def test_inspect_rejects_identifier_file(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/inspect",
        files=upload_files(identifier_csv()),
        data={"deidentified_confirmed": "true"},
    )
    assert response.status_code == 422
    detail = response.json()["error"]
    assert "identifier" in detail["message"].lower()
    assert {item["column"] for item in detail["identifier_columns"]} >= {"full_name", "email"}


def test_inspect_rejects_non_csv(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/inspect",
        files={"file": ("cohort.txt", io.BytesIO(safe_csv(3)), "text/plain")},
        data={"deidentified_confirmed": "true"},
    )
    assert response.status_code == 422


def test_analyse_requires_explicit_confirmation(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(5)),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "false"},
    )
    assert response.status_code == 400
    assert "screening report" in response.json()["error"]


def test_analyse_returns_aggregates_and_rows(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(30)),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    aggregate = body["aggregate"]
    assert aggregate["rows_scored"] == 30
    assert 0 <= aggregate["reference_percentile"]["median"] <= 100
    assert aggregate["deviation_score"]["min"] >= 0
    assert len(body["rows"]) == 30
    assert body["clustering"]["available"] is False
    assert body["persistence"].startswith("none")
    # No disease-probability or cancer-type claim may appear in the payload.
    serialised = str(body).lower()
    for banned in (
        "probability of cancer",
        "cancer risk",
        "risk score",
        "predicted cancer",
        "likely cancer",
        "diagnosis of",
    ):
        assert banned not in serialised


def test_analyse_excludes_leakage_columns_from_features(authed_client):
    response = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(leakage_csv()),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    )
    assert response.status_code == 200
    used = response.json()["aggregate"]["features_used"]
    assert "Cancer" not in used and "tcga_stage_ordinal" not in used


def test_analyse_rejects_file_with_no_usable_features(authed_client):
    payload = b"row_id,colour,shape\n1,red,round\n2,blue,square\n"
    response = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(payload),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    )
    assert response.status_code == 422
    assert "cannot be analysed" in response.json()["error"]["message"]


def test_upload_is_never_persisted(authed_client, tmp_path):
    before = set(p.name for p in config.APP_ROOT.rglob("*.csv"))
    authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(10), name="secret_cohort.csv"),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    )
    after = set(p.name for p in config.APP_ROOT.rglob("*.csv"))
    assert before == after
    assert "secret_cohort.csv" not in after


def test_row_cap_limits_scored_rows(authed_client, monkeypatch):
    monkeypatch.setattr(config, "MAX_SCORED_ROWS", 5)
    response = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(20)),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    )
    body = response.json()
    assert body["aggregate"]["rows_scored"] == 5
    assert body["aggregate"]["rows_accepted"] == 20
    assert body["aggregate"]["row_cap_applied"] is True


def test_oversize_upload_returns_413(authed_client):
    payload = b"DEMO_RIDAGEYR\n" + b"55\n" * 10
    payload += b"#" * (config.MAX_UPLOAD_BYTES + 10)
    response = authed_client.post(
        "/api/v1/dataset/inspect",
        files=upload_files(payload),
        data={"deidentified_confirmed": "true"},
    )
    assert response.status_code == 413


def test_results_csv_export(authed_client):
    analysis = authed_client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(8)),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    ).json()
    response = authed_client.post("/api/v1/dataset/export", json={"rows": analysis["rows"]})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("row_number,source_row_index,metabolic_deviation_score")
    assert len(lines) == 9
    assert lines[1].endswith(",metabolic_deviation_and_representation,no")


def test_export_rejects_empty_payload(authed_client):
    assert authed_client.post("/api/v1/dataset/export", json={"rows": []}).status_code == 422
