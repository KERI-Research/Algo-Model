"""Access-key authentication and signed-cookie sessions.

Design notes
------------
* The professor types a plaintext access key. The server hashes it with SHA-256
  and compares the hex digest to ``METABOGUARD_ACCESS_KEY_SHA256`` using
  :func:`hmac.compare_digest` (constant time). The plaintext is never stored,
  never logged, never echoed back and never written to disk.
* A successful login issues a signed session token
  (``payload.signature``, HMAC-SHA256 over the payload with
  ``METABOGUARD_SESSION_SECRET``). There is no server-side session store, so
  nothing about a session is persisted anywhere.
* The token is delivered as a ``__Host-`` prefixed, HttpOnly, Secure,
  SameSite=Strict cookie on path ``/`` with no Domain attribute.
* The same token is also returned in the login response body so the dashboard
  can hold it in memory and send it as ``Authorization: Bearer`` when the
  browser refuses cookies (the thread-preview iframe blocks cookie storage).
  It is kept in React state only - never localStorage, sessionStorage or
  IndexedDB - and it carries no secret material beyond the session signature.
* Failures are deliberately generic: the caller cannot tell an unknown key from
  a malformed one, and cannot tell an expired session from a forged one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, Response

from . import config

GENERIC_LOGIN_FAILURE = "Invalid access key."
GENERIC_RATE_LIMITED = "Too many attempts. Try again later."
GENERIC_UNAUTHORISED = "Authentication required."
AUTH_NOT_CONFIGURED = "Server access control is not configured."


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_access_key(plaintext: str) -> str:
    """SHA-256 hex digest of a plaintext access key."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_access_key(plaintext: str) -> bool:
    """Constant-time comparison of the SHA-256 digest against the configured hash.

    Both operands are always the same length (64 hex characters), so no timing
    signal leaks from the comparison itself. When the server is unconfigured we
    still perform a comparison against a dummy digest so that the unconfigured
    path takes the same shape as the configured one.
    """
    configured = config.access_key_hash()
    candidate = hash_access_key(plaintext or "")
    reference = configured or hashlib.sha256(os.urandom(32)).hexdigest()
    matched = hmac.compare_digest(candidate, reference)
    return bool(configured) and matched


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------


def issue_session_token(now: float | None = None) -> tuple[str, int]:
    """Return ``(token, expires_at_epoch_seconds)``."""
    secret = config.session_secret()
    if secret is None:
        raise HTTPException(status_code=503, detail=AUTH_NOT_CONFIGURED)
    issued_at = int(now if now is not None else time.time())
    expires_at = issued_at + config.SESSION_TTL_SECONDS
    payload = {
        "sub": "metaboguard-professor",
        "iat": issued_at,
        "exp": expires_at,
        "jti": _b64encode(os.urandom(12)),
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64encode(hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}", expires_at


def verify_session_token(token: str | None, now: float | None = None) -> dict[str, Any] | None:
    """Validate signature and expiry. Returns the payload, or None when invalid."""
    secret = config.session_secret()
    if not token or secret is None or token.count(".") != 1:
        return None
    encoded, signature = token.split(".", 1)
    expected = _b64encode(hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_b64decode(encoded))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expires_at = payload.get("exp")
    if not isinstance(expires_at, int):
        return None
    current = now if now is not None else time.time()
    if current >= expires_at:
        return None
    return payload


def set_session_cookie(response: Response, token: str) -> None:
    """Attach the session cookie with the attributes the deployment requires."""
    response.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=token,
        max_age=config.SESSION_TTL_SECONDS,
        expires=config.SESSION_TTL_SECONDS,
        path="/",
        domain=None,  # __Host- prefix forbids a Domain attribute.
        secure=True,
        httponly=True,
        samesite="strict",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=config.SESSION_COOKIE_NAME,
        path="/",
        domain=None,
        secure=True,
        httponly=True,
        samesite="strict",
    )


def extract_token(request: Request) -> str | None:
    """Prefer the cookie; fall back to a bearer token for cookie-blocked hosts."""
    cookie = request.cookies.get(config.SESSION_COOKIE_NAME)
    if cookie:
        return cookie
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


def current_session(request: Request) -> dict[str, Any] | None:
    return verify_session_token(extract_token(request))


def require_session(request: Request) -> dict[str, Any]:
    """FastAPI dependency guarding every model, dataset and research route."""
    payload = current_session(request)
    if payload is None:
        raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORISED)
    return payload


# ---------------------------------------------------------------------------
# Login rate limiting (in-memory, per process, nothing persisted)
# ---------------------------------------------------------------------------


@dataclass
class LoginRateLimiter:
    max_attempts: int = config.LOGIN_MAX_ATTEMPTS
    window_seconds: int = config.LOGIN_WINDOW_SECONDS
    lockout_seconds: int = config.LOGIN_LOCKOUT_SECONDS
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)

    def _prune(self, key: str, now: float) -> list[float]:
        recent = [t for t in self._failures.get(key, []) if now - t < self.window_seconds]
        self._failures[key] = recent
        return recent

    def retry_after(self, key: str, now: float | None = None) -> int:
        """Seconds the caller must wait, or 0 when a login attempt is allowed."""
        now = now if now is not None else time.time()
        until = self._locked_until.get(key, 0.0)
        if until > now:
            return int(until - now) + 1
        if until:
            self._locked_until.pop(key, None)
            self._failures.pop(key, None)
        return 0

    def register_failure(self, key: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        recent = self._prune(key, now)
        recent.append(now)
        if len(recent) >= self.max_attempts:
            self._locked_until[key] = now + self.lockout_seconds
            self._failures[key] = []

    def register_success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)

    def reset(self) -> None:
        self._failures.clear()
        self._locked_until.clear()


login_rate_limiter = LoginRateLimiter()


def client_fingerprint(request: Request) -> str:
    """Coarse client identity for rate limiting. Not logged, not persisted."""
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    host = forwarded or (request.client.host if request.client else "unknown")
    return hashlib.sha256(host.encode("utf-8")).hexdigest()[:32]
