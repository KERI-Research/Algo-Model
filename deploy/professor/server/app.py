"""MetaboGuard professor dashboard - FastAPI application.

Route posture
-------------
Simulation-only future-risk routes live under ``/api/v1/simulation/*`` and are
session-protected like everything else. The clinical future-risk route exists
only to refuse: it always returns 409.

Unauthenticated: ``/api/v1/health``, ``/api/v1/status``, ``/api/v1/session``
(reports whether a session exists), ``/api/v1/auth/login``, ``/api/v1/auth/logout``.
Everything else - model metadata, patient probe, dataset intake, dataset
analysis, results export, reliability, clustering, evidence - requires a valid
signed session, enforced server side so that bypassing the frontend route guard
achieves nothing.

There is no telemetry, no analytics, no outbound network call and no persistent
server state. Request bodies are never logged.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, config, dataset, future_risk, model, reports

app = FastAPI(
    title="MetaboGuard Professor Dashboard",
    description=(
        "Access-controlled research review surface for the MetaboGuard self-supervised "
        "metabolic deviation model. Non-diagnostic. Developed within the KERI department."
    ),
    version=config.DEPLOYMENT_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Same-origin in every deployment target (static assets and API share the host).
# Credentials are allowed so the __Host- session cookie is sent; no wildcard origin
# is used, because a wildcard is incompatible with credentialed requests.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://[a-z0-9-]+\.pplx\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
    max_age=600,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    access_key: str = Field(min_length=1, max_length=512)


class ProbeRequest(BaseModel):
    patient_record: dict[str, Any]
    confirm_explicit_scoring: bool = False


class SimulationScoreRequest(BaseModel):
    """A synthetic longitudinal history. No identifiers are accepted."""

    visits: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    simulation_mode: bool = False
    seed: int | None = Field(default=None, ge=1, le=999_999)
    archetype: Literal[
        "reference_range",
        "metabolic_deviation",
        "reported_diabetes_metabolic",
        "sparse_but_valid",
    ] | None = None


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "metaboguard-professor-dashboard"}


@app.get("/api/v1/status")
async def status() -> dict[str, Any]:
    """Deployment posture. Contains no secret, no hash and no private data."""
    return {
        "status": "ok",
        "deployment_version": config.DEPLOYMENT_VERSION,
        "model_version": config.MODEL_VERSION,
        "auth_configured": config.auth_configured(),
        "session_ttl_hours": config.SESSION_TTL_SECONDS // 3600,
        "posture": "non_diagnostic_research_discovery",
        "persistence": "none",
        "telemetry": "none",
        "upload_limits": {
            "max_megabytes": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_rows": config.MAX_UPLOAD_ROWS,
            "max_scored_rows": config.MAX_SCORED_ROWS,
            "accepted_types": ["text/csv"],
        },
    }


@app.get("/api/v1/session")
async def session_status(request: Request) -> dict[str, Any]:
    payload = auth.current_session(request)
    if payload is None:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "expires_at": payload.get("exp"),
        "seconds_remaining": max(0, int(payload.get("exp", 0) - time.time())),
    }


@app.post("/api/v1/auth/login")
async def login(request: Request, payload: LoginRequest, response: Response) -> dict[str, Any]:
    if not config.auth_configured():
        raise HTTPException(status_code=503, detail=auth.AUTH_NOT_CONFIGURED)
    fingerprint = auth.client_fingerprint(request)
    retry_after = auth.login_rate_limiter.retry_after(fingerprint)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=auth.GENERIC_RATE_LIMITED,
            headers={"Retry-After": str(retry_after)},
        )
    if not auth.verify_access_key(payload.access_key):
        auth.login_rate_limiter.register_failure(fingerprint)
        raise HTTPException(status_code=401, detail=auth.GENERIC_LOGIN_FAILURE)
    auth.login_rate_limiter.register_success(fingerprint)
    token, expires_at = auth.issue_session_token()
    auth.set_session_cookie(response, token)
    return {
        "authenticated": True,
        "expires_at": expires_at,
        # Held in memory by the client for hosts that block cookie storage.
        "session_token": token,
        "token_transport_note": (
            "The session is a signed cookie. This copy exists only for embedded contexts "
            "where the browser blocks cookies; keep it in memory, never in storage."
        ),
    }


@app.post("/api/v1/auth/logout")
async def logout(response: Response) -> dict[str, Any]:
    auth.clear_session_cookie(response)
    return {"authenticated": False, "message": "Session ended."}


# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------


@app.get("/api/v1/model")
async def model_route(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    return model.model_summary()


@app.get("/api/v1/overview")
async def overview_route(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    reliability = reports.reliability_report()
    clustering = reports.clustering_report("complete_cases")
    integrity = reports.integrity_report()
    model_summary = model.model_summary()
    tiers = reliability.get("tiers") or {}
    return {
        "posture": {
            "headline": "Non-diagnostic discovery research",
            "statement": (
                "MetaboGuard looks for unusual metabolic profiles in de-identified research "
                "data. It does not detect, diagnose, stage or predict cancer or diabetes, and "
                "every output requires clinician interpretation."
            ),
            "intended_reader": "Research supervisor and clinical collaborators.",
        },
        "model": model_summary,
        "status_cards": [
            {
                "id": "inference",
                "label": "Inference path",
                "value": "PyTorch to NumPy",
                "state": "ok",
                "detail": (
                    f"Selected offline with {model_summary['training_backend']}; exported "
                    "weights replayed by NumPy. No training in deployment."
                ),
            },
            {
                "id": "reliability",
                "label": "Reliability audit",
                "value": str(reliability.get("status", "unknown")),
                "state": "ok" if reliability.get("status") == "ok" else "warn",
                "detail": (
                    f"{len(tiers.get('usable_now', []))} features usable now, "
                    f"{len(tiers.get('qualified_use', []))} qualified, "
                    f"{len(tiers.get('unavailable', []))} unavailable."
                ),
            },
            {
                "id": "clustering",
                "label": "Phenotype clustering",
                "value": str(clustering.get("status", "unknown")),
                "state": "abstain",
                "detail": "Every candidate was dominated by the survey-cycle negative control.",
            },
            {
                "id": "horizon",
                "label": "Future-risk surfaces",
                "value": "simulation only",
                "state": "warn",
                "detail": (
                    "Clinical scoring returns 409. Selected synthetic-data horizons are "
                    "available only in the simulation panel."
                ),
            },
        ],
        "data_limitations": [
            {
                "id": "cross_sectional",
                "title": "Cross-sectional data only",
                "detail": (
                    "The training data hold one observation per participant, so nothing in this "
                    "deployment can describe change over time or future onset."
                ),
            },
            {
                "id": "prevalent_labels",
                "title": "Labels are prevalent, not incident",
                "detail": (
                    "Any recorded diagnosis was already present at the time of measurement, so "
                    "an association cannot be read as early detection."
                ),
            },
            {
                "id": "survey_cycle",
                "title": "Survey-cycle effects dominate structure",
                "detail": (
                    "Assay methods and availability changed between NHANES cycles, and that "
                    "signal outweighs metabolic structure in unsupervised analysis."
                ),
            },
            {
                "id": "weights",
                "title": "Survey weights are not applied",
                "detail": (
                    "Outputs describe the analytic sample, not the US adult population."
                ),
            },
            {
                "id": "external",
                "title": "No external validation",
                "detail": (
                    "There is no independent cohort, no calibration study and no prospective "
                    "specimen collection meeting the PRoBE design."
                ),
            },
        ],
        "integrity": {
            "status": integrity.get("status"),
            "horizon_gates": integrity.get("horizon_gates"),
            "row_counts": integrity.get("row_counts"),
        },
        "non_diagnostic_warning": model.NON_DIAGNOSTIC_WARNING,
    }


@app.post("/api/v1/probe/score")
async def probe_score(
    payload: ProbeRequest, _: dict = Depends(auth.require_session)
) -> dict[str, Any]:
    if not payload.confirm_explicit_scoring:
        raise HTTPException(
            status_code=400,
            detail="Scoring must be requested explicitly for each record.",
        )
    try:
        return model.score_single_record(payload.patient_record)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.get("/api/v1/probe/schema")
async def probe_schema(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    metadata = model.artifact_metadata()
    return {
        "features": metadata.get("features", []),
        "categorical_features": dataset.CATEGORICAL_FEATURES,
        "plausible_ranges": {
            key: list(value) for key, value in dataset.PLAUSIBLE_RANGES.items()
        },
        "field_meanings": model.FIELD_MEANINGS,
        "evidence_boundaries": model.EVIDENCE_BOUNDARIES,
    }


async def _read_upload(file: UploadFile) -> bytearray:
    """Stream an upload into memory with a hard byte ceiling."""
    buffer = bytearray()
    while True:
        chunk = await file.read(1024 * 256)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > config.MAX_UPLOAD_BYTES:
            dataset.shred_buffer(buffer)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                    "upload limit."
                ),
            )
    return buffer


@app.post("/api/v1/dataset/inspect")
async def dataset_inspect(
    file: UploadFile = File(...),
    deidentified_confirmed: bool = Form(False),
    _: dict = Depends(auth.require_session),
) -> dict[str, Any]:
    if not deidentified_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Confirm that the file contains de-identified research data before upload.",
        )
    buffer: bytearray | None = None
    frame = None
    try:
        buffer = await _read_upload(file)
        frame = dataset.read_csv_bytes(buffer, file.filename or "upload.csv")
        report = dataset.build_intake_report(frame, file.filename or "upload.csv")
        report["requires_confirmation_before_analysis"] = True
        report["persistence"] = "none: parsed in memory and discarded with this response."
        return report
    except dataset.DatasetRejected as error:
        raise HTTPException(
            status_code=422, detail={"message": error.reason, **error.detail}
        )
    finally:
        dataset.shred_buffer(buffer)
        del frame
        await file.close()


@app.post("/api/v1/dataset/analyse")
async def dataset_analyse(
    file: UploadFile = File(...),
    deidentified_confirmed: bool = Form(False),
    analysis_confirmed: bool = Form(False),
    include_rows: bool = Form(True),
    _: dict = Depends(auth.require_session),
) -> dict[str, Any]:
    if not deidentified_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Confirm that the file contains de-identified research data before upload.",
        )
    if not analysis_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Confirm the screening report before running model analysis.",
        )
    started_at = time.monotonic()
    buffer: bytearray | None = None
    frame = None
    try:
        buffer = await _read_upload(file)
        frame = dataset.read_csv_bytes(buffer, file.filename or "upload.csv")
        intake = dataset.build_intake_report(frame, file.filename or "upload.csv")
        if not intake["model_ready"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "This file cannot be analysed by the model.",
                    "blockers": intake["blockers"],
                },
            )
        usable = [
            feature
            for feature in intake["schema"]["mapped_features"]
            if intake["feature_eligibility"][feature]["tier"] in {"usable_now", "qualified_use"}
        ]
        mask = dataset.eligible_row_mask(frame, usable)
        analysis = model.score_dataset(frame, usable, mask, started_at=started_at)
        if dataset.analysis_budget_exceeded(started_at):
            analysis["aggregate"]["time_budget_note"] = (
                "Analysis exceeded the soft compute budget; reduce the row count for faster "
                "results."
            )
        if not include_rows:
            analysis["rows"] = []
        analysis["intake"] = intake
        return analysis
    except dataset.DatasetRejected as error:
        raise HTTPException(
            status_code=422, detail={"message": error.reason, **error.detail}
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    finally:
        dataset.shred_buffer(buffer)
        del frame
        await file.close()


class ExportRequest(BaseModel):
    rows: list[dict[str, Any]]


@app.post("/api/v1/dataset/export")
async def dataset_export(
    payload: ExportRequest, _: dict = Depends(auth.require_session)
) -> PlainTextResponse:
    """Render results the client already holds as CSV. Nothing is stored server side."""
    if not payload.rows:
        raise HTTPException(status_code=422, detail="No result rows were supplied.")
    if len(payload.rows) > config.MAX_SCORED_ROWS:
        raise HTTPException(status_code=413, detail="Too many result rows to export.")
    body = model.results_csv(payload.rows)
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="metaboguard_research_results.csv"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/v1/simulation/capability")
async def simulation_capability(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    """What the simulation-only future-risk models can and cannot report."""
    return future_risk.capability()


@app.post("/api/v1/simulation/score")
async def simulation_score(
    payload: SimulationScoreRequest, _: dict = Depends(auth.require_session)
) -> dict[str, Any]:
    """Score a synthetic longitudinal history. Simulation only, nothing retained."""
    identifier_fields = future_risk.reject_identifier_fields(payload.visits)
    if identifier_fields:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Visit rows may only contain simulated measurements. Remove any "
                    "identifier-like field before submitting."
                ),
                "rejected_fields": identifier_fields,
            },
        )
    try:
        return future_risk.score_history(payload.visits, payload.simulation_mode)
    except future_risk.SimulationInputRejected as error:
        raise HTTPException(
            status_code=422, detail={"message": error.reason, **error.detail}
        )


@app.post("/api/v1/future-risk/score")
async def clinical_future_risk(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    """Clinical future-risk scoring is permanently refused in this deployment."""
    raise HTTPException(
        status_code=409,
        detail={
            "message": future_risk.CLINICAL_ENDPOINT_MESSAGE,
            "capability_state": "simulation_only_longitudinal",
            "use_instead": "/api/v1/simulation/score",
        },
    )


@app.get("/api/v1/reliability")
async def reliability_route(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    return reports.reliability_report()


@app.get("/api/v1/clusters")
async def clusters_route(
    variant: str = "complete_cases", _: dict = Depends(auth.require_session)
) -> dict[str, Any]:
    try:
        return reports.clustering_report(variant)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown clustering variant '{variant}'.",
                "available_variants": ["complete_cases", "all_adults"],
            },
        )


@app.get("/api/v1/evidence")
async def evidence_route(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    return reports.evidence_payload()


@app.get("/api/v1/integrity")
async def integrity_route(_: dict = Depends(auth.require_session)) -> dict[str, Any]:
    return reports.integrity_report()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Uniform error envelope. No stack traces and no request bodies are exposed."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
        headers=exc.headers or {},
    )


# ---------------------------------------------------------------------------
# Static frontend (served only when a build is present)
# ---------------------------------------------------------------------------

if config.STATIC_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=config.STATIC_DIR / "assets", check_dir=False),
        name="static-assets",
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(config.STATIC_DIR / "index.html")

    @app.get("/{path:path}")
    async def spa_fallback(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Unknown endpoint.")
        candidate = (config.STATIC_DIR / path).resolve()
        if (
            candidate.is_file()
            and config.STATIC_DIR.resolve() in candidate.parents
        ):
            return FileResponse(candidate)
        return FileResponse(config.STATIC_DIR / "index.html")
