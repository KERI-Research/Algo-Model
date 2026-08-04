"""Every model, dataset and research route must refuse an unauthenticated caller."""

from __future__ import annotations

import pytest
from conftest import safe_csv, upload_files

PUBLIC_ROUTES = [
    ("GET", "/api/v1/health"),
    ("GET", "/api/v1/status"),
    ("GET", "/api/v1/session"),
]

PROTECTED_GET_ROUTES = [
    "/api/v1/model",
    "/api/v1/overview",
    "/api/v1/probe/schema",
    "/api/v1/reliability",
    "/api/v1/clusters",
    "/api/v1/evidence",
    "/api/v1/integrity",
]


@pytest.mark.parametrize("method,path", PUBLIC_ROUTES)
def test_public_routes_do_not_require_a_session(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 200


@pytest.mark.parametrize("path", PROTECTED_GET_ROUTES)
def test_protected_get_routes_reject_anonymous_callers(client, path):
    assert client.get(path).status_code == 401


def test_protected_post_routes_reject_anonymous_callers(client):
    assert client.post(
        "/api/v1/probe/score",
        json={"patient_record": {"DEMO_RIDAGEYR": 50}, "confirm_explicit_scoring": True},
    ).status_code == 401
    assert client.post(
        "/api/v1/dataset/inspect",
        files=upload_files(safe_csv(5)),
        data={"deidentified_confirmed": "true"},
    ).status_code == 401
    assert client.post(
        "/api/v1/dataset/analyse",
        files=upload_files(safe_csv(5)),
        data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
    ).status_code == 401
    assert client.post("/api/v1/dataset/export", json={"rows": []}).status_code == 401


def test_status_exposes_no_secret_material(client):
    body = client.get("/api/v1/status").json()
    serialised = str(body)
    assert body["auth_configured"] is True
    assert "SHA256" not in serialised
    assert "secret" not in serialised.lower()
    assert body["persistence"] == "none"
    assert body["telemetry"] == "none"


def test_unknown_api_path_is_not_served_as_html(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert "text/html" not in response.headers.get("content-type", "")


def test_security_headers_present(client):
    response = client.get("/api/v1/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
