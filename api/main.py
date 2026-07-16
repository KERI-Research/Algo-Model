"""
Causal Inference API Router
===========================
This file serves as the communication bridge between your robust React frontend and the heavy Python data processing engine. Adopting a forward-thinking view, we utilize FastAPI to ensure high performance and asynchronous request handling. Always maintain a skeptical approach to your incoming data payloads. Keep up the phenomenal work—you are building a truly innovative multi-tier architecture that thinks outside the standard predictive modeling box.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import CausalExecutionError, execute_pipeline
from predictive import execute_predictive_baseline
from pathlib import Path
from fetch_nhanes import ensure_nhanes_dataset
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    dataset: str = "nhanes_merged.csv"
    treatment: str = "Diabetes"
    outcome: str = "Cancer"
    allow_fallback: bool = True


class DatasetRequest(BaseModel):
    dataset: str = "nhanes_merged.csv"


def resolve_dataset_path(dataset_name: str) -> Path | None:
    """Resolve dataset file across common project locations."""
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

    if Path(dataset_name).name == "nhanes_merged.csv":
        ensure_nhanes_dataset()
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
            dataset_paths.setdefault(csv_path.name, csv_path)

    if not dataset_paths:
        ensure_nhanes_dataset()

        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                continue

            for csv_path in dataset_dir.glob("*.csv"):
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