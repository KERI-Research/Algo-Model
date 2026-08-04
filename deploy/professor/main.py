"""Production entrypoint: serves the FastAPI app (and the built client) on port 5000."""

from __future__ import annotations

import os

import uvicorn

from server.app import app  # noqa: F401  (imported for `uvicorn main:app`)

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        log_level="warning",
        access_log=False,  # No request logging: no upload or session data in logs.
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
