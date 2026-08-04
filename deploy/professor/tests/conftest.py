"""Shared fixtures. Test secrets are generated per session and never persisted."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_ACCESS_KEY = "professor-test-key-do-not-reuse"
TEST_ACCESS_KEY_SHA256 = hashlib.sha256(TEST_ACCESS_KEY.encode()).hexdigest()
TEST_SESSION_SECRET = "unit-test-session-secret-not-a-real-secret-value"


@pytest.fixture(autouse=True)
def configured_env(monkeypatch):
    from server import auth, config

    monkeypatch.setenv(config.ACCESS_KEY_HASH_ENV, TEST_ACCESS_KEY_SHA256)
    monkeypatch.setenv(config.SESSION_SECRET_ENV, TEST_SESSION_SECRET)
    auth.login_rate_limiter.reset()
    yield
    auth.login_rate_limiter.reset()


@pytest.fixture
def client():
    from server.app import app

    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture
def authed_client(client):
    response = client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    assert response.status_code == 200
    return client


SAFE_CSV_HEADER = (
    "row_id,DEMO_RIDAGEYR,DEMO_RIAGENDR,BMX_BMXBMI,BMX_BMXWAIST,GHB_LBXGH,GLU_LBXGLU,"
    "INS_LBXIN,TRIGLY_LBXTR,HDL_LBDHDD,TCHOL_LBXTC,HSCRP_LBXHSCRP,smoking_status,homa_ir\n"
)


def safe_csv(rows: int = 40) -> bytes:
    lines = [SAFE_CSV_HEADER]
    for index in range(rows):
        age = 30 + index % 45
        bmi = 22 + (index % 18)
        hba1c = round(5.0 + (index % 12) * 0.2, 1)
        glucose = 80 + (index % 40)
        insulin = round(4 + (index % 20) * 0.7, 1)
        lines.append(
            f"r{index},{age},{1 + index % 2},{bmi},{75 + index % 30},{hba1c},{glucose},"
            f"{insulin},{90 + index % 60},{45 + index % 20},{170 + index % 40},"
            f"{round(0.5 + (index % 9) * 0.4, 1)},{index % 3},"
            f"{round(insulin * glucose / 405, 2)}\n"
        )
    return "".join(lines).encode()


def identifier_csv() -> bytes:
    header = "full_name,email,DEMO_RIDAGEYR,BMX_BMXBMI,GHB_LBXGH\n"
    rows = "".join(
        f"Person {i},person{i}@example.org,{40 + i},{25 + i},{5.4 + i * 0.1}\n" for i in range(5)
    )
    return (header + rows).encode()


def leakage_csv() -> bytes:
    header = (
        "row_id,DEMO_RIDAGEYR,DEMO_RIAGENDR,BMX_BMXBMI,GHB_LBXGH,GLU_LBXGLU,INS_LBXIN,"
        "Cancer,PancreaticCancer,tcga_stage_ordinal\n"
    )
    rows = "".join(
        f"r{i},{45 + i},{1 + i % 2},{27 + i},{5.6 + i * 0.1},{95 + i},{9.5 + i},"
        f"{i % 2},0,{i % 4}\n"
        for i in range(12)
    )
    return (header + rows).encode()


def upload_files(payload: bytes, name: str = "cohort.csv"):
    return {"file": (name, io.BytesIO(payload), "text/csv")}
