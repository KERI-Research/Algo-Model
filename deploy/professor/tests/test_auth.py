"""Authentication, session and rate-limiting tests."""

from __future__ import annotations

import hashlib
import inspect
import time

from conftest import TEST_ACCESS_KEY, TEST_ACCESS_KEY_SHA256

from server import auth, config


def test_hash_access_key_is_sha256():
    assert auth.hash_access_key(TEST_ACCESS_KEY) == TEST_ACCESS_KEY_SHA256
    assert auth.hash_access_key("abc") == hashlib.sha256(b"abc").hexdigest()


def test_verify_access_key_accepts_correct_key(configured_env):
    assert auth.verify_access_key(TEST_ACCESS_KEY) is True


def test_verify_access_key_rejects_wrong_key(configured_env):
    assert auth.verify_access_key("wrong-key") is False
    assert auth.verify_access_key("") is False


def test_verify_access_key_uses_constant_time_comparison():
    """The comparison must go through hmac.compare_digest on fixed-length digests."""
    source = inspect.getsource(auth.verify_access_key)
    assert "hmac.compare_digest" in source
    assert "==" not in source.split("compare_digest")[0].split("candidate")[-1]
    assert len(auth.hash_access_key("a")) == len(auth.hash_access_key("a" * 5000)) == 64


def test_verify_access_key_fails_closed_when_unconfigured(monkeypatch):
    monkeypatch.delenv(config.ACCESS_KEY_HASH_ENV, raising=False)
    assert auth.verify_access_key(TEST_ACCESS_KEY) is False


def test_login_success_sets_host_prefixed_cookie(client):
    response = client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    raw = response.headers["set-cookie"]
    assert raw.startswith("__Host-metaboguard-session=")
    assert "HttpOnly" in raw
    assert "Secure" in raw
    assert "SameSite=strict" in raw or "SameSite=Strict" in raw
    assert "Path=/" in raw
    assert "Domain=" not in raw
    assert f"Max-Age={config.SESSION_TTL_SECONDS}" in raw
    assert config.SESSION_TTL_SECONDS == 8 * 60 * 60


def test_login_response_never_contains_secrets(client):
    response = client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    body = response.text
    assert TEST_ACCESS_KEY not in body
    assert TEST_ACCESS_KEY_SHA256 not in body
    assert "SESSION_SECRET" not in body
    assert "unit-test-session-secret" not in body


def test_login_failure_is_generic(client):
    wrong = client.post("/api/v1/auth/login", json={"access_key": "not-the-key"}).json()
    malformed = client.post("/api/v1/auth/login", json={"access_key": " "}).json()
    assert wrong["error"] == auth.GENERIC_LOGIN_FAILURE
    assert malformed["error"] == auth.GENERIC_LOGIN_FAILURE
    assert "hash" not in str(wrong).lower()


def test_login_rate_limited_after_repeated_failures(client):
    for _ in range(config.LOGIN_MAX_ATTEMPTS):
        assert client.post("/api/v1/auth/login", json={"access_key": "bad"}).status_code == 401
    blocked = client.post("/api/v1/auth/login", json={"access_key": "bad"})
    assert blocked.status_code == 429
    assert blocked.json()["error"] == auth.GENERIC_RATE_LIMITED
    assert int(blocked.headers["retry-after"]) > 0
    # A correct key is refused too while the lockout stands (fail closed).
    assert client.post(
        "/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY}
    ).status_code == 429


def test_rate_limiter_window_and_reset():
    limiter = auth.LoginRateLimiter(max_attempts=3, window_seconds=60, lockout_seconds=120)
    now = 1000.0
    for offset in range(2):
        limiter.register_failure("client", now=now + offset)
    assert limiter.retry_after("client", now=now + 3) == 0
    limiter.register_failure("client", now=now + 3)
    assert limiter.retry_after("client", now=now + 4) > 0
    assert limiter.retry_after("client", now=now + 200) == 0
    limiter.register_failure("client", now=now + 300)
    limiter.register_success("client")
    assert limiter.retry_after("client", now=now + 301) == 0


def test_session_token_roundtrip_and_expiry(configured_env):
    token, expires_at = auth.issue_session_token(now=1_000_000)
    assert expires_at == 1_000_000 + config.SESSION_TTL_SECONDS
    assert auth.verify_session_token(token, now=1_000_100) is not None
    assert auth.verify_session_token(token, now=expires_at) is None
    assert auth.verify_session_token(token, now=expires_at + 1) is None


def test_session_token_rejects_tampering(configured_env):
    token, _ = auth.issue_session_token()
    payload, signature = token.split(".")
    assert auth.verify_session_token(f"{payload}x.{signature}") is None
    assert auth.verify_session_token(f"{payload}.{signature[:-2]}aa") is None
    assert auth.verify_session_token("garbage") is None
    assert auth.verify_session_token(None) is None


def test_session_token_invalid_under_a_different_secret(monkeypatch):
    token, _ = auth.issue_session_token()
    monkeypatch.setenv(config.SESSION_SECRET_ENV, "a-completely-different-secret-value")
    assert auth.verify_session_token(token) is None


def test_session_status_and_logout(client):
    assert client.get("/api/v1/session").json() == {"authenticated": False}
    client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    status = client.get("/api/v1/session").json()
    assert status["authenticated"] is True
    assert 0 < status["seconds_remaining"] <= config.SESSION_TTL_SECONDS
    logout = client.post("/api/v1/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/v1/session").json() == {"authenticated": False}
    assert client.get("/api/v1/model").status_code == 401


def test_bearer_token_transport_is_accepted(client):
    token = client.post(
        "/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY}
    ).json()["session_token"]
    client.cookies.clear()
    assert client.get("/api/v1/model").status_code == 401
    ok = client.get("/api/v1/model", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200


def test_expired_session_is_rejected(client, monkeypatch):
    token, _ = auth.issue_session_token(now=time.time() - config.SESSION_TTL_SECONDS - 10)
    response = client.get("/api/v1/model", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json()["error"] == auth.GENERIC_UNAUTHORISED


def test_login_unavailable_when_server_unconfigured(client, monkeypatch):
    monkeypatch.delenv(config.SESSION_SECRET_ENV, raising=False)
    response = client.post("/api/v1/auth/login", json={"access_key": TEST_ACCESS_KEY})
    assert response.status_code == 503
    assert response.json()["error"] == auth.AUTH_NOT_CONFIGURED


def test_client_fingerprint_is_hashed(client):
    class _Request:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        client = type("C", (), {"host": "10.0.0.1"})()

    fingerprint = auth.client_fingerprint(_Request())
    assert "203.0.113.7" not in fingerprint
    assert len(fingerprint) == 32
