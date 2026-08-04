"""MetaboGuard API router (FastAPI).

Bridges the React dashboard to the Python analysis modules. Terminology in this
API is deliberately strict:

* ``metabolic_deviation_score`` - how unusual a metabolic profile is versus the
  training reference distribution. Not a probability of anything.
* ``reference_percentile`` - rank of that deviation score in the reference.
* ``latent_representation`` - the learned encoding of the input features.
* ``cross_sectional_association`` - association with an ALREADY-recorded
  diagnosis in a cross-sectional survey.

None of these is a future disease probability. Future-horizon endpoints stay
fail-closed until a horizon passes the event-count safety gate
(``data_integrity.horizon_gate_report``).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from biomarker import execute_biomarker_discovery
from engine import CausalExecutionError, execute_pipeline
from predictive import execute_predictive_baseline
from self_supervised import dataset_capabilities, score_records
from data_integrity import (
    DEFAULT_HORIZON_DAYS,
    INVALIDATED_DATASETS,
    INVALIDATED_TARGETS,
    MIN_EVENTS_PER_HORIZON,
    MIN_NON_EVENTS_PER_HORIZON,
    horizon_gate_report,
    validate_dataset,
)
import json
from pathlib import Path
from fetch_nhanes import ensure_nhanes_dataset
from fetch_tcga import ensure_tcga_cdr_dataset
import pandas as pd

app = FastAPI(
    title="MetaboGuard API",
    description=(
        "Non-diagnostic metabolic research API. On the current cross-sectional data "
        "it returns metabolic DEVIATION scores, reference PERCENTILES, latent "
        "REPRESENTATIONS and CROSS-SECTIONAL ASSOCIATIONS with already-recorded "
        "diagnoses. It does not return future disease probabilities. Clinician review "
        "is required. Developed within the KERI department."
    ),
    version="1.1.0",
)

NON_DIAGNOSTIC_WARNING = (
    "Research use only, non-diagnostic. Outputs are metabolic deviation scores, "
    "percentiles, latent representations or cross-sectional associations with "
    "already-recorded diagnoses. They are not diagnoses and not future disease "
    "probabilities. Clinician review is required."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEPRECATED_DATASETS = {
    "nhanes_merged.csv": (
        "Deprecated: pre-Priority-A schema and incorrect pancreatic code. "
        "Use nhanes_merged_v2.csv or nhanes_multicycle_v2.csv."
    ),
    "nhanes_multicycle.csv": (
        "Deprecated: this file used MCQ230 code 39 ('Other') as pancreatic "
        "cancer. Use nhanes_multicycle_v2.csv, which uses the official code 29."
    )
}

class AnalysisRequest(BaseModel):
    dataset: str = "nhanes_multicycle_v2.csv"
    treatment: str = "Diabetes"
    outcome: str = "Cancer"
    allow_fallback: bool = True


class DatasetRequest(BaseModel):
    dataset: str = "nhanes_multicycle_v2.csv"


class BiomarkerRequest(BaseModel):
    dataset: str = "nhanes_multicycle_v2.csv"
    patient_record: dict[str, object] | None = None
    top_k: int = 8
    force_retrain: bool = False
    # Which label to predict. Default 'Cancer' preserves backward compatibility.
    # NHANES also supports 'PancreaticCancer'. TCGA also supports 'Progression'.
    target: str = "Cancer"
    # Optional cohort filter, e.g. 'diabetics_only' for pancreatic risk
    # stratification within diabetic patients (per the MetaboGuard brief).
    cohort_filter: str | None = None


class PreventionScoreRequest(BaseModel):
    patient_record: dict[str, object]
    artifact: str = "nhanes_multicycle_v2"


class PreventionCapabilitiesRequest(BaseModel):
    dataset: str = "nhanes_multicycle_v2.csv"


def resolve_dataset_path(dataset_name: str) -> Path | None:
    """Resolve dataset file across common project locations."""
    if Path(dataset_name).name in DEPRECATED_DATASETS:
        return None
    api_dir = Path(__file__).resolve().parent
    project_root = api_dir.parent

    candidates = [
        Path(dataset_name),
        project_root / "data" / dataset_name,
        api_dir / "nhanes_data" / dataset_name,
        project_root / "nhanes_data" / dataset_name,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    dataset_leaf = Path(dataset_name).name
    if dataset_leaf == "nhanes_merged_v2.csv":
        ensure_nhanes_dataset()
    elif dataset_leaf == "tcga_cdr.csv":
        ensure_tcga_cdr_dataset()
    else:
        return None

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    return None


def list_available_datasets() -> list[dict[str, str]]:
    api_dir = Path(__file__).resolve().parent
    project_root = api_dir.parent

    dataset_dirs = [project_root / "data", api_dir / "nhanes_data", project_root / "nhanes_data"]
    dataset_paths: dict[str, Path] = {}

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue

        for csv_path in dataset_dir.glob("*.csv"):
            if csv_path.name in DEPRECATED_DATASETS:
                continue
            dataset_paths.setdefault(csv_path.name, csv_path)

    # Ensure both first-party datasets are materialized so they show up in the
    # picker even on a fresh clone.
    ensure_nhanes_dataset()
    try:
        ensure_tcga_cdr_dataset()
    except Exception as error:
        print(f"[warn] Could not materialize tcga_cdr.csv: {error}")

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue

        for csv_path in dataset_dir.glob("*.csv"):
            if csv_path.name in DEPRECATED_DATASETS:
                continue
            dataset_paths.setdefault(csv_path.name, csv_path)

    return [
        {
            "name": dataset_name,
            "path": str(dataset_path),
        }
        for dataset_name, dataset_path in sorted(dataset_paths.items())
    ]


def preview_dataset(file_path: Path, sample_size: int = 5) -> dict[str, object]:
    dataframe = pd.read_csv(file_path, nrows=sample_size)
    return {
        "dataset": file_path.name,
        "columns": list(dataframe.columns),
        "preview": dataframe.fillna("").to_dict(orient="records"),
        "sample_size": len(dataframe),
    }


@app.get("/api/v1/datasets")
async def get_datasets():
    return {"datasets": list_available_datasets()}


@app.post("/api/v1/dataset-preview")
async def get_dataset_preview(request: DatasetRequest):
    file_path = resolve_dataset_path(request.dataset)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset not found. Expected one of: "
                f"data/{request.dataset}, api/nhanes_data/{request.dataset}, or nhanes_data/{request.dataset}"
            ),
        )

    try:
        return preview_dataset(file_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/api/v1/analyze")
async def analyze_data(request: AnalysisRequest):
    file_path = resolve_dataset_path(request.dataset)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset not found. Expected one of: "
                f"data/{request.dataset}, api/nhanes_data/{request.dataset}, or nhanes_data/{request.dataset}"
            ),
        )
        
    try:
        results = execute_pipeline(
            str(file_path),
            treatment=request.treatment,
            outcome=request.outcome,
            allow_fallback=request.allow_fallback,
        )
        return results
    except CausalExecutionError as error:
        if error.fallback_result is not None:
            fallback_payload = {
                **error.fallback_result,
                "warnings": [
                    *(error.fallback_result.get("warnings", [])),
                    "Strict causal mode failed; returning fallback estimate.",
                ],
                "strict_mode_requested": True,
                "strict_mode_failed": True,
            }
            return JSONResponse(status_code=200, content=fallback_payload)

        raise HTTPException(
            status_code=422,
            detail={
                "message": str(error),
                "fallback_preview": error.fallback_result,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predictive-baseline")
async def predictive_baseline(request: DatasetRequest):
    file_path = resolve_dataset_path(request.dataset)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset not found. Expected one of: "
                f"data/{request.dataset}, api/nhanes_data/{request.dataset}, or nhanes_data/{request.dataset}"
            ),
        )

    try:
        return execute_predictive_baseline(str(file_path))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/v1/biomarker-discovery")
async def biomarker_discovery(request: BiomarkerRequest):
    file_path = resolve_dataset_path(request.dataset)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Dataset not found. Expected one of: "
                f"data/{request.dataset}, api/nhanes_data/{request.dataset}, or nhanes_data/{request.dataset}"
            ),
        )

    try:
        return execute_biomarker_discovery(
            str(file_path),
            patient_record=request.patient_record,
            top_k=request.top_k,
            force_retrain=request.force_retrain,
            target=request.target,
            cohort_filter=request.cohort_filter,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/v1/prevention-capabilities")
async def prevention_capabilities(request: PreventionCapabilitiesRequest):
    file_path = resolve_dataset_path(request.dataset)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or invalidated.")
    frame = pd.read_csv(file_path, low_memory=False)
    gates = horizon_gate_report(frame, DEFAULT_HORIZON_DAYS)
    return {
        "project": "MetaboGuard",
        "intended_use": "Preventive research and clinician-reviewed early warning.",
        "not_intended_for": "Diagnosis, treatment or patient reassurance.",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "capabilities": dataset_capabilities(frame),
        "future_horizon_gates": gates,
        "longitudinal_heads_enabled": bool(gates["any_horizon_eligible"]),
        "gate_policy": (
            f"A horizon is only enabled with at least {MIN_EVENTS_PER_HORIZON} events "
            f"and {MIN_NON_EVENTS_PER_HORIZON} non-events."
        ),
        "invalidated_datasets": INVALIDATED_DATASETS,
        "invalidated_supervised_targets": INVALIDATED_TARGETS,
    }


@app.post("/api/v1/data-integrity")
async def data_integrity(request: PreventionCapabilitiesRequest):
    """Machine-readable data-integrity report (coding, leakage, gates, splits)."""
    file_path = resolve_dataset_path(request.dataset)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or invalidated.")
    try:
        report = validate_dataset(file_path, strict=False)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return report.as_dict()


def _resolve_ssl_artifact(name: str) -> Path:
    """Resolve a trained SSL artifact directory, preferring the promoted pointer."""
    root = Path(__file__).resolve().parent.parent / "model_artifacts" / "metaboguard_ssl"
    if name in {"current", "CURRENT"}:
        pointer = root / "CURRENT.json"
        if not pointer.exists():
            raise HTTPException(
                status_code=409,
                detail="No promoted artifact: run train_self_supervised.py --promote first.",
            )
        return Path(json.loads(pointer.read_text())["artifact_dir"])
    candidate = root / name
    if (candidate / "metadata.json").exists():
        return candidate
    run_candidate = root / "runs" / name
    if (run_candidate / "metadata.json").exists():
        return run_candidate
    raise HTTPException(
        status_code=409,
        detail=(
            "Self-supervised artifact is not trained. Run "
            "train_self_supervised.py first. No diagnostic fallback is used."
        ),
    )


@app.post("/api/v1/prevention-score")
async def prevention_score(request: PreventionScoreRequest):
    artifact = _resolve_ssl_artifact(request.artifact)
    metadata = json.loads((artifact / "metadata.json").read_text())
    result = score_records(pd.DataFrame([request.patient_record]), artifact)[0]
    warnings: list[str] = []
    if metadata.get("run_label") == "smoke":
        warnings.append(
            "This artifact came from a bounded SMOKE run and is for demonstration only."
        )
    if metadata.get("capabilities", {}).get("supports_future_development_prediction"):
        warnings.append("Longitudinal capability reported: verify gates before use.")
    return {
        "project": "MetaboGuard",
        "output_type": "metabolic_deviation_and_representation",
        "is_future_risk_probability": False,
        "score": result,
        "field_meanings": {
            "metabolic_deviation_score": "How unusual this profile is versus the training reference.",
            "reference_percentile": "Rank of the deviation score within the training reference distribution.",
            "latent_representation": f"{metadata.get('latent_dim', 16)}-dimensional learned encoding.",
            "top_deviation_features": "Features contributing most to reconstruction error.",
        },
        "artifact": {
            "dir": str(artifact),
            "model_name": metadata.get("model_name"),
            "run_label": metadata.get("run_label"),
            "code_version": metadata.get("code_version"),
            "backend": metadata.get("backend"),
            "created_at": metadata.get("created_at"),
            "dataset_sha256": (metadata.get("dataset_fingerprint") or {}).get("sha256"),
        },
        "warnings": warnings,
        "clinical_warning": NON_DIAGNOSTIC_WARNING,
    }


@app.post("/api/v1/prevention-future-risk")
async def prevention_future_risk(request: PreventionCapabilitiesRequest):
    """Future-horizon head. Intentionally fail-closed until capability gates pass."""
    file_path = resolve_dataset_path(request.dataset)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or invalidated.")
    frame = pd.read_csv(file_path, low_memory=False)
    gates = horizon_gate_report(frame, DEFAULT_HORIZON_DAYS)
    raise HTTPException(
        status_code=409,
        detail={
            "message": (
                "Future-risk (1/3/5-year) scoring is disabled. The current data are "
                "cross-sectional, so no horizon passes the event-count safety gate."
            ),
            "intended_horizons_days": list(DEFAULT_HORIZON_DAYS),
            "gate": gates,
            "blocker": (
                "Patient-level longitudinal follow-up with incident outcomes is required "
                "(NHANES here has one observation per participant; TCGA is post-diagnosis)."
            ),
        },
    )
