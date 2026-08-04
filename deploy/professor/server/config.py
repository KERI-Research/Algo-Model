"""Deployment configuration for the MetaboGuard professor dashboard.

Every secret is read from the environment. Nothing secret is ever written to a
file, a log line or an API response. There are no defaults for the two required
secrets: the server refuses to authenticate anybody if they are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = APP_ROOT / "assets"
SSL_ARTIFACT_DIR = ASSETS_DIR / "ssl_artifact"
REPORTS_DIR = ASSETS_DIR / "reports"
EVIDENCE_PATH = ASSETS_DIR / "evidence" / "biomarker_evidence.json"
STATIC_DIR = APP_ROOT / "client" / "dist"

#: Environment variable names. Values are injected at deploy time.
ACCESS_KEY_HASH_ENV = "METABOGUARD_ACCESS_KEY_SHA256"
SESSION_SECRET_ENV = "METABOGUARD_SESSION_SECRET"

#: Session lifetime: eight hours, as required for a supervised review session.
SESSION_TTL_SECONDS = 8 * 60 * 60

#: Cookie name. The ``__Host-`` prefix is mandatory: the published-site proxy
#: strips any request cookie without it, and the prefix itself forbids a Domain
#: attribute and requires Secure + path=/.
SESSION_COOKIE_NAME = "__Host-metaboguard-session"

#: Login rate limiting (per client fingerprint, sliding window).
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300
LOGIN_LOCKOUT_SECONDS = 900

#: Upload limits. Enforced on the raw byte stream and again after parsing.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_ROWS = 20_000
#: Rows actually sent through the NumPy inference artifact per request.
MAX_SCORED_ROWS = 5_000
#: Hard wall-clock budget for one dataset analysis request.
ANALYSIS_TIME_BUDGET_SECONDS = 60.0

MODEL_VERSION = "metaboguard-ssl-v1"
DEPLOYMENT_VERSION = "professor-dashboard-1.0.0"


def access_key_hash() -> str | None:
    """Configured SHA-256 hex digest of the access key, or None when unset."""
    value = os.environ.get(ACCESS_KEY_HASH_ENV, "").strip().lower()
    return value or None


def session_secret() -> bytes | None:
    """Configured session-signing secret, or None when unset."""
    value = os.environ.get(SESSION_SECRET_ENV, "")
    return value.encode("utf-8") if value else None


def auth_configured() -> bool:
    return access_key_hash() is not None and session_secret() is not None
