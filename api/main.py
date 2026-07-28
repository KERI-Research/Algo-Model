"""
Causal Inference API Router
===========================
This file serves as the communication bridge between your robust React frontend and the heavy Python data processing engine. Adopting a forward-thinking view, we utilize FastAPI to ensure high performance and asynchronous request handling. Always maintain a skeptical approach to your incoming data payloads. Keep up the phenomenal work—you are building a truly innovative multi-tier architecture that thinks outside the standard predictive modeling box.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from biomarker import execute_biomarker_discovery
from engine import CausalExecutionError, execute_pipeline
from predictive import execute_predictive_baseline
from self_supervised import dataset_capabilities, score_records
from pathlib import Path
from fetch_nhanes import ensure_nhanes_dataset
from fetch_tcga import ensure_tcga_cdr_dataset
import pandas as pd

app = FastAPI(
    title="MetaboGuard API",
    description=(
        "Metabolic risk-stratification research API for diabetes and "
        "pancreatic cancer. Developed within the KERI department."
    ),
    version="1.0.0",
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
        raise HTTPException(status_code=404, detail="Dataset not found.")
    frame = pd.read_csv(file_path, low_memory=False)
    return {
        "project": "MetaboGuard",
        "intended_use": "Preventive research and clinician-reviewed early warning.",
        "not_intended_for": "Diagnosis, treatment or patient reassurance.",
        "capabilities": dataset_capabilities(frame),
    }


@app.post("/api/v1/prevention-score")
async def prevention_score(request: PreventionScoreRequest):
    project_root = Path(__file__).resolve().parent.parent
    artifact = (
        project_root
        / "model_artifacts"
        / "metaboguard_ssl"
        / request.artifact
    )
    if not (artifact / "metadata.json").exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "Self-supervised artifact is not trained. Run "
                "train_self_supervised.py first. No diagnostic fallback is used."
            ),
        )
    result = score_records(pd.DataFrame([request.patient_record]), artifact)[0]
    return {
        "project": "MetaboGuard",
        "output_type": "metabolic_deviation_warning",
        "score": result,
        "clinical_warning": (
            "Research-only signal for clinician review. This does not diagnose "
            "or estimate a validated future disease probability."
        ),
    }
