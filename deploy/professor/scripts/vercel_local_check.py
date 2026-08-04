#!/usr/bin/env python3
"""Serverless-equivalent local check of the Vercel path configuration.

What this emulates
------------------
* the ``vercel.json`` rewrites: ``/api/v1/:path*`` -> the ``api/index.py``
  function, everything else -> the static build, with ``index.html`` as the SPA
  fallback (filesystem first, exactly like Vercel);
* the trimmed function runtime: scikit-learn, SciPy, joblib and PyTorch are
  blocked at import time, so only what the root ``requirements.txt`` installs is
  available.

What it asserts
---------------
static asset and SPA routing, login (correct and incorrect key), `__Host-`
cookie attributes, protected-route rejection, rate limiting, patient probe
scoring, CSV upload screening, identifier rejection, dataset analysis, CSV
export, logout, and that nothing was written to disk.

Usage (from the project root, after `npm --prefix client run build:vercel`):

    METABOGUARD_ACCESS_KEY_SHA256=$(printf '%s' key | sha256sum | cut -d' ' -f1) \
    METABOGUARD_SESSION_SECRET=local-check-secret \
    python3 scripts/vercel_local_check.py key
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sys
from pathlib import Path

BLOCKED = {"sklearn", "scipy", "joblib", "torch", "self_supervised", "data_reliability"}


class _ImportBlocker:
    """Emulate the trimmed serverless environment."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"not installed in the Vercel runtime: {name}")
        return None

    def find_module(self, name, path=None):  # pragma: no cover - legacy hook
        return self.find_spec(name, path)


sys.meta_path.insert(0, _ImportBlocker())

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACCESS_KEY = sys.argv[1] if len(sys.argv) > 1 else "local-check-key"
os.environ.setdefault(
    "METABOGUARD_ACCESS_KEY_SHA256", hashlib.sha256(ACCESS_KEY.encode()).hexdigest()
)
# Generated per run: no secret value is ever written into this file.
os.environ.setdefault("METABOGUARD_SESSION_SECRET", secrets.token_urlsafe(48))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import FileResponse, Response  # noqa: E402
from starlette.routing import Route  # noqa: E402

from api.index import app as function_app  # noqa: E402
from server import config, core_bridge  # noqa: E402

DIST = ROOT / "client" / "dist"
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {label}{f' - {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(label)


async def static_or_spa(request):
    """Filesystem first, then index.html - the vercel.json rewrite order."""
    relative = request.path_params["path"].lstrip("/")
    if relative.startswith("api/"):
        return Response("not found", status_code=404)
    candidate = (DIST / relative).resolve()
    if relative and candidate.is_file() and DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(DIST / "index.html")


static_app = Starlette(routes=[Route("/{path:path}", static_or_spa)])


async def emulated(scope, receive, send):
    """Dispatch like Vercel: /api/v1/* to the function (path preserved), else static.

    A Vercel rewrite does not strip the matched prefix, so the function receives
    the original ``/api/v1/...`` path - which is exactly what the FastAPI routes
    are registered under.
    """
    if scope["type"] == "http" and scope["path"].startswith("/api/v1"):
        await function_app(scope, receive, send)
        return
    await static_app(scope, receive, send)


def main() -> int:
    if not (DIST / "index.html").exists():
        print("client/dist is missing: run `npm --prefix client run build:vercel` first.")
        return 1

    print("runtime profile")
    check("scikit-learn/SciPy/joblib blocked", not (BLOCKED & set(sys.modules)))
    check("bridge uses exported constants", core_bridge.VENDORED_MODULES_AVAILABLE is False)

    bundle = next((DIST / "assets").glob("index-*.js")).read_text()
    print("static build")
    if "__PORT_5000__" in bundle:
        print("  client/dist was produced by the pplx build (`npm run build`).")
        print("  Run `npm --prefix client run build:vercel` before this check.")
    check("no pplx proxy sentinel in the bundle", "__PORT_5000__" not in bundle)
    check("no secret material in the bundle",
          "METABOGUARD_SESSION_SECRET" not in bundle and ACCESS_KEY not in bundle)
    check("api calls are same-origin /api/v1", '"/api/v1/auth/login"' in bundle
          or "/api/v1/auth/login" in bundle)

    with TestClient(emulated, base_url="https://metaboguard.test") as client:
        print("routing")
        index = client.get("/")
        check("GET / serves the SPA shell", index.status_code == 200
              and "<div id=\"root\">" in index.text)
        asset_name = next((DIST / "assets").glob("index-*.js")).name
        asset = client.get(f"/assets/{asset_name}")
        check("hashed static asset is served", asset.status_code == 200
              and "javascript" in asset.headers["content-type"])
        spa = client.get("/dataset")
        check("unknown path falls back to index.html", spa.status_code == 200
              and "<div id=\"root\">" in spa.text)
        health = client.get("/api/v1/health")
        check("GET /api/v1/health reaches the function", health.status_code == 200
              and health.json()["status"] == "ok")
        status = client.get("/api/v1/status").json()
        check("status reports auth configured", status["auth_configured"] is True)
        check("status leaks no secret", "secret" not in str(status).lower())

        print("authentication")
        check("protected route rejects anonymous", client.get("/api/v1/model").status_code == 401)
        bad = client.post("/api/v1/auth/login", json={"access_key": "wrong-key"})
        check("wrong key -> generic 401", bad.status_code == 401
              and bad.json()["error"] == "Invalid access key.")
        good = client.post("/api/v1/auth/login", json={"access_key": ACCESS_KEY})
        cookie = good.headers.get("set-cookie", "")
        check("correct key -> 200", good.status_code == 200)
        check("__Host- cookie with the required attributes",
              cookie.startswith("__Host-metaboguard-session=")
              and "HttpOnly" in cookie and "Secure" in cookie
              and "samesite=strict" in cookie.lower() and "Path=/" in cookie
              and "Domain=" not in cookie)
        check("login response contains no key or hash",
              ACCESS_KEY not in good.text
              and hashlib.sha256(ACCESS_KEY.encode()).hexdigest() not in good.text)
        model = client.get("/api/v1/model")
        check("protected route accepts the session", model.status_code == 200)
        check("model reports the NumPy/exported path",
              model.json()["inference_backend"] == "numpy"
              and model.json()["preprocessor_path"] == "exported_constants")

        print("model surfaces")
        probe = client.post(
            "/api/v1/probe/score",
            json={
                "patient_record": {
                    "DEMO_RIDAGEYR": 61, "BMX_BMXBMI": 33.4, "GHB_LBXGH": 6.4,
                    "GLU_LBXGLU": 128, "INS_LBXIN": 21.5, "homa_ir": 6.8,
                },
                "confirm_explicit_scoring": True,
            },
        )
        score = probe.json()["score"]
        check("patient probe scores explicitly", probe.status_code == 200
              and score["metabolic_deviation_score"] > 0
              and 0 <= score["reference_percentile"] <= 100,
              f"score={score['metabolic_deviation_score']} pct={score['reference_percentile']}")
        check("probe refuses implicit scoring", client.post(
            "/api/v1/probe/score",
            json={"patient_record": {"DEMO_RIDAGEYR": 50}, "confirm_explicit_scoring": False},
        ).status_code == 400)
        for path in ("/api/v1/overview", "/api/v1/reliability", "/api/v1/clusters",
                     "/api/v1/evidence", "/api/v1/integrity", "/api/v1/probe/schema"):
            check(f"GET {path}", client.get(path).status_code == 200)

        print("dataset upload")
        fixtures = ROOT / "fixtures"
        safe = (fixtures / "safe_deidentified_cohort.csv").read_bytes()
        identifiers = (fixtures / "identifier_cohort_REJECT.csv").read_bytes()
        unconfirmed = client.post(
            "/api/v1/dataset/inspect",
            files={"file": ("cohort.csv", safe, "text/csv")},
            data={"deidentified_confirmed": "false"},
        )
        check("upload without the de-identification tick -> 400", unconfirmed.status_code == 400)
        rejected = client.post(
            "/api/v1/dataset/inspect",
            files={"file": ("cohort.csv", identifiers, "text/csv")},
            data={"deidentified_confirmed": "true"},
        )
        check("identifier fixture rejected", rejected.status_code == 422
              and len(rejected.json()["error"]["identifier_columns"]) >= 4)
        intake = client.post(
            "/api/v1/dataset/inspect",
            files={"file": ("cohort.csv", safe, "text/csv")},
            data={"deidentified_confirmed": "true"},
        )
        body = intake.json()
        check("safe fixture screened", intake.status_code == 200
              and body["rows"]["accepted"] == 238 and body["model_ready"] is True,
              f"accepted={body.get('rows', {}).get('accepted')}")
        check("analysis requires confirmation", client.post(
            "/api/v1/dataset/analyse",
            files={"file": ("cohort.csv", safe, "text/csv")},
            data={"deidentified_confirmed": "true", "analysis_confirmed": "false"},
        ).status_code == 400)
        analysis = client.post(
            "/api/v1/dataset/analyse",
            files={"file": ("cohort.csv", safe, "text/csv")},
            data={"deidentified_confirmed": "true", "analysis_confirmed": "true"},
        )
        aggregate = analysis.json()["aggregate"]
        check("dataset scored in memory", analysis.status_code == 200
              and aggregate["rows_scored"] == 238,
              f"median_pct={aggregate['reference_percentile']['median']}")
        check("clustering stays unavailable", analysis.json()["clustering"]["available"] is False)
        export = client.post(
            "/api/v1/dataset/export", json={"rows": analysis.json()["rows"]}
        )
        lines = export.text.strip().splitlines()
        check("results CSV exported", export.status_code == 200
              and export.headers["content-type"].startswith("text/csv")
              and len(lines) == 239)

        print("session lifecycle")
        check("logout clears the session", client.post("/api/v1/auth/logout").status_code == 200)
        check("protected route rejects after logout",
              client.get("/api/v1/model").status_code == 401)
        forged = client.get("/api/v1/model", headers={"Authorization": "Bearer forged.token"})
        check("forged bearer token rejected", forged.status_code == 401)
        statuses = [
            client.post("/api/v1/auth/login", json={"access_key": "wrong"}).status_code
            for _ in range(6)
        ]
        check("login rate limiting engages", statuses[-1] == 429, f"{statuses}")

    print("persistence")
    stray = [
        path.name
        for path in config.APP_ROOT.rglob("*.csv")
        if "fixtures" not in path.parts and "node_modules" not in path.parts
    ]
    check("no uploaded CSV written to disk", not stray, str(stray))
    check("no database file created", not list(config.APP_ROOT.rglob("*.db")))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all Vercel-path checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
