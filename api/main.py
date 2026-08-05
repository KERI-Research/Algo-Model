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
from research_contract import (
    FUTURE_RISK_DISABLED_STATEMENT,
    NON_DIAGNOSTIC_WARNING,
    PROFILE_WARNING_STATEMENT,
    SECTION_TITLES,
    capability_state_from_supported_output,
    deviation_band_from_percentile,
)
import json
from pathlib import Path
from typing import Any
from typing import Any
from typing import Any
from typing import Any
from typing import Any
from typing import Any
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
        frame = pd.read_csv(file_path, low_memory=False)
        capabilities = dataset_capabilities(frame)
        dataset_capability_state = capability_state_from_supported_output(
            capabilities.get("supported_output")
        )
        return execute_biomarker_discovery(
            str(file_path),
            patient_record=request.patient_record,
            top_k=request.top_k,
            force_retrain=request.force_retrain,
            target=request.target,
            cohort_filter=request.cohort_filter,
            dataset_capability_state=dataset_capability_state,
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
    capabilities = dataset_capabilities(frame)
    return {
        "project": "MetaboGuard",
        "intended_use": "Preventive research and clinician-reviewed early warning.",
        "not_intended_for": "Diagnosis, treatment or patient reassurance.",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "capabilities": capabilities,
        "dataset_capability_state": capability_state_from_supported_output(
            capabilities.get("supported_output")
        ),
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
    source_dataset = metadata.get("source_dataset") or "nhanes_multicycle_v2.csv"
    source_path = resolve_dataset_path(source_dataset)
    if source_path and source_path.exists():
        capability_frame = pd.read_csv(source_path, low_memory=False)
        capabilities = dataset_capabilities(capability_frame)
    else:
        capabilities = {
            "supported_output": "cross_sectional_representation_and_deviation_only",
            "supports_future_development_prediction": False,
        }
    dataset_capability_state = capability_state_from_supported_output(
        capabilities.get("supported_output")
    )
    result = score_records(pd.DataFrame([request.patient_record]), artifact)[0]
    deviation_band = deviation_band_from_percentile(result.get("reference_percentile"))
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
        "dataset_capability_state": dataset_capability_state,
        "score": result,
        "patient_assessment": {
            "current_profile_assessment": {
                "section_title": SECTION_TITLES["current_profile_assessment"],
                "deviation_band": deviation_band["key"],
                "deviation_band_label": deviation_band["label"],
                "reference_percentile": result.get("reference_percentile"),
                "warning_label": deviation_band["label"],
                "note": PROFILE_WARNING_STATEMENT,
            },
            "standout_factors": {
                "section_title": SECTION_TITLES["standout_factors"],
                "top_deviation_features": result.get("top_deviation_features", []),
            },
            "data_readiness": {
                "section_title": SECTION_TITLES["data_readiness"],
                "missing_fields": [
                    {
                        "field": field,
                        "priority": "medium",
                        "why_it_matters": "Additional inputs improve profile interpretation breadth.",
                        "expected_impact_bucket": "moderate_confidence_gain",
                    }
                    for field in metadata.get("features", [])
                    if field not in request.patient_record
                ],
                "dataset_capability_state": dataset_capability_state,
            },
            "research_association": {
                "section_title": SECTION_TITLES["research_association"],
                "status": "disabled_on_this_route",
                "note": FUTURE_RISK_DISABLED_STATEMENT,
            },
            "safety_contract": {
                "diagnostic_status": "non_diagnostic",
                "future_risk": "disabled",
                "clinical_warning": "profile_deviation_only",
            },
        },
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

# ---------------------------------------------------------------------------
# Research surfaces (reliability, evidence provenance, exploratory phenotypes)
# ---------------------------------------------------------------------------

EXPLANATION_CLASSES = {
    "data_observation": "Measured directly in the data file (counts, coverage, drift).",
    "model_association": "Produced by our model on our sample. Not validated, not causal.",
    "published_evidence": "From a catalogued source with URL, study design and evidence grade.",
    "causal_claim_not_established": "Default status for any mechanism statement.",
}


class ResearchClusterRequest(BaseModel):
    run: str = "latest"
    variant: str = "complete_cases"


@app.post("/api/v1/data-reliability")
async def data_reliability_route(request: PreventionCapabilitiesRequest):
    """Structured reliability audit with feature eligibility tiers (data observation)."""
    from data_reliability import build_reliability_report

    file_path = resolve_dataset_path(request.dataset)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or invalidated.")
    try:
        report = build_reliability_report(file_path, strict=False).as_dict()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {
        "explanation_class": "data_observation",
        "explanation_classes": EXPLANATION_CLASSES,
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "status": report["status"],
        "dataset": report["dataset"],
        "tier_definitions": report["tier_definitions"],
        "tiers": report["tiers"],
        "feature_eligibility": report["feature_eligibility"],
        "violations": report["violations"],
        "sections": {
            key: report["sections"][key]
            for key in (
                "provenance",
                "row_counts",
                "label_confidence",
                "survey_weights",
                "capability_state",
                "leakage_controls",
            )
        },
        "assay_cycle_drift_summary": {
            "level_drift_features": report["sections"]["assay_cycle_drift"].get(
                "level_drift_features"
            ),
            "availability_gap_features": report["sections"]["assay_cycle_drift"].get(
                "availability_gap_features"
            ),
            "note": report["sections"]["assay_cycle_drift"].get("note"),
        },
    }


@app.get("/api/v1/evidence-catalogue")
async def evidence_catalogue_route():
    """Biomarker evidence with mandatory provenance (published evidence)."""
    from evidence_catalogue import load_catalogue

    try:
        catalogue = load_catalogue()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error))
    summary = catalogue.summary()
    return {
        "explanation_class": "published_evidence",
        "explanation_classes": EXPLANATION_CLASSES,
        "causal_status": "causal_claim_not_established",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "summary": summary,
        "policy": catalogue.policy,
        "clinician_ready_entries": catalogue.doctor_facing_entries(),
        "research_only_entries": [
            {
                "entry_id": entry["entry_id"],
                "marker_or_panel": entry["marker_or_panel"],
                "reason": "Missing source URL/DOI or ungraded evidence.",
            }
            for entry in catalogue.research_only_entries()
        ],
        "panel_framing": summary["panel_framing"],
    }


def _resolve_research_run(run: str) -> Path:
    root = Path(__file__).resolve().parent.parent / "model_artifacts" / "research_runs"
    if not root.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "No research run found. Run api/run_research_pass.py first. "
                "No placeholder cluster output is served."
            ),
        )
    if run not in {"latest", "LATEST"}:
        candidate = root / run
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Research run '{run}' not found.")
        return candidate
    runs = sorted(path for path in root.glob("research__*") if path.is_dir())
    if not runs:
        raise HTTPException(status_code=409, detail="No research run directories present.")
    return runs[-1]


@app.post("/api/v1/research-clusters")
async def research_clusters(request: ResearchClusterRequest):
    """Exploratory phenotype clusters. Never a cancer type, never future risk."""
    run_dir = _resolve_research_run(request.run)
    report_path = run_dir / f"clustering_{request.variant}" / "clustering_report.json"
    if not report_path.exists():
        available = sorted(
            path.name.replace("clustering_", "")
            for path in run_dir.glob("clustering_*")
            if path.is_dir()
        )
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Variant '{request.variant}' not present in {run_dir.name}.",
                "available_variants": available,
            },
        )
    report = json.loads(report_path.read_text())
    response = {
        "explanation_class": "model_association",
        "explanation_classes": EXPLANATION_CLASSES,
        "causal_status": "causal_claim_not_established",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "run": run_dir.name,
        "variant": request.variant,
        "status": report["status"],
        "output_type": report["output_type"],
        "is_disease_classification": False,
        "labels_used_in_fit_or_selection": report["labels_used_in_fit_or_selection"],
        "cluster_naming_policy": (
            "Clusters are patient/metabolic phenotypes. Labelling a cluster as a cancer "
            "type, cancer site or disease subtype is prohibited."
        ),
        "space": report["space"],
        "split_source": report["split_source"],
        "warnings": report["warnings"],
        "candidate_summary": [
            {
                "method": item["method"],
                "k": item["k"],
                "silhouette": item.get("train_metrics", {}).get("silhouette"),
                "davies_bouldin": item.get("train_metrics", {}).get("davies_bouldin"),
                "calinski_harabasz": item.get("train_metrics", {}).get("calinski_harabasz"),
                "bootstrap_mean_ari": item.get("bootstrap_stability", {}).get("mean_ari"),
                "seed_mean_ari": item.get("seed_stability", {}).get("mean_ari"),
                "negative_controls": item.get("negative_controls", {}).get("controls"),
                "dominated_by": item.get("negative_controls", {}).get("dominated_by"),
                "passes_gates": item.get("passes_gates"),
                "gate_failures": item.get("gate_failures"),
            }
            for item in report["candidates"]
            if item.get("status") == "evaluated"
        ],
    }
    if report["status"] == "no_stable_clusters":
        response["abstain"] = {
            "reason": report["abstain_reason"],
            "gate_failure_summary": report.get("gate_failure_summary"),
            "interpretation": (
                "No phenotype solution is reported because none passed the stability and "
                "negative-control gates. This is a result, not a failure to produce output."
            ),
        }
    else:
        response["selected"] = report["selected"]
        response["clusters"] = report["characterisation"]["clusters"]
        response["membership_confidence"] = report["characterisation"]["membership_confidence"]
        response["posthoc_label_summary"] = report["characterisation"]["posthoc_label_summary"]
        response["panel_framing"] = report["characterisation"]["panel_framing"]
    return response

# ---------------------------------------------------------------------------
# Future risk: capability reporting and SIMULATION-ONLY scoring
# ---------------------------------------------------------------------------


class SimulationRiskRequest(BaseModel):
    simulation_mode: bool = False
    artifact: str = "latest"
    outcome: str = "type2_diabetes"
    #: Longitudinal history: list of visits, each {"days_before_index": int, features...}
    patient_history: list[dict[str, Any]] = []
    #: Explicitly reject cross-sectional payloads for future-risk scoring.
    patient_record: dict[str, Any] | None = None


def _future_risk_artifact(name: str) -> Path:
    root = Path(__file__).resolve().parent.parent / "model_artifacts" / "future_risk"
    if not root.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "No future-risk artifact exists. Run api/run_future_risk_pipeline.py on a "
                "simulated cohort first. No fallback risk is produced."
            ),
        )
    if name not in {"latest", "LATEST"}:
        candidate = root / name / "artifact"
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Future-risk artifact '{name}' not found.")
        return candidate
    # A run only counts when its artifact actually carries metadata; placeholder or partial
    # directories must not be resolved as "latest".
    runs = sorted(
        path
        for path in root.glob("simulation*")
        if (path / "artifact" / "metadata.json").exists()
    )
    if not runs:
        raise HTTPException(status_code=409, detail="No completed future-risk run found.")
    return runs[-1] / "artifact"


@app.get("/api/v1/future-risk-capability")
async def future_risk_capability():
    """What future-risk output is possible right now, and why."""
    from longitudinal_schema import (
        CAPABILITY_PERMISSIONS,
        CapabilityState,
        DISABLED_OUTCOMES,
        HORIZON_DAYS,
        MIN_EVENTS_PER_HORIZON,
    )

    artifact_state = None
    try:
        artifact = _future_risk_artifact("latest")
        metadata = json.loads((artifact / "metadata.json").read_text())
        artifact_state = {
            "artifact_dir": str(artifact),
            "capability_state": metadata["capability_state"],
            "simulation_only": metadata["simulation_only"],
            "clinical_use": metadata["clinical_use"],
            "outcomes_trained": metadata["outcomes_trained"],
            "selection": metadata["selection"],
            "created_at": metadata["created_at"],
        }
    except HTTPException as error:
        artifact_state = {"available": False, "reason": error.detail}

    return {
        "explanation_class": "model_association",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
        "clinical_future_risk_enabled": False,
        "clinical_future_risk_blocker": (
            "No real longitudinal cohort with incident outcomes exists in this repository. "
            "The clinical endpoint /api/v1/prevention-future-risk therefore returns HTTP 409."
        ),
        "simulated_future_risk_enabled": True,
        "simulated_future_risk_requires": "simulation_mode=true plus a simulation-only artifact",
        "capability_states": {
            state.value: CAPABILITY_PERMISSIONS[state] for state in CapabilityState
        },
        "horizons_days": list(HORIZON_DAYS),
        "event_gate": {"minimum_events": MIN_EVENTS_PER_HORIZON, "minimum_non_events": MIN_EVENTS_PER_HORIZON},
        "disabled_outcomes": DISABLED_OUTCOMES,
        "artifact": artifact_state,
        "simulation_banner": (
            "Simulated risk is produced from synthetic data for software verification only. "
            "It is not validated for patient risk and must never be shown as a patient's risk."
        ),
    }


@app.post("/api/v1/simulation/future-risk-score")
async def simulation_future_risk_score(request: SimulationRiskRequest):
    """Score a synthetic longitudinal history. Simulation only, double-gated."""
    import joblib
    import numpy as np
    import pandas as pd

    from longitudinal_schema import (
        HORIZON_DAYS,
        HORIZON_LABELS,
        PREVENTION_SAFE_FEATURES,
        assert_outcome_allowed,
        assert_simulated_future_risk_allowed,
    )

    if request.patient_record is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Cross-sectional records cannot be scored for future risk. Supply "
                "patient_history as a list of visits with days_before_index."
            ),
        )
    if not request.simulation_mode:
        raise HTTPException(
            status_code=403,
            detail=(
                "This endpoint is simulation only and requires an explicit simulation_mode=true "
                "flag. Real patient future-risk scoring stays disabled (HTTP 409)."
            ),
        )
    if not request.patient_history:
        raise HTTPException(status_code=422, detail="patient_history is required and must be longitudinal.")
    if len({visit.get("days_before_index") for visit in request.patient_history}) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least two distinct visit times are required; a single visit is cross-sectional.",
        )

    artifact = _future_risk_artifact(request.artifact)
    metadata = json.loads((artifact / "metadata.json").read_text())
    try:
        assert_simulated_future_risk_allowed(metadata["capability_state"], request.simulation_mode)
        assert_outcome_allowed(request.outcome)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error))
    if request.outcome not in metadata["outcomes_trained"]:
        raise HTTPException(
            status_code=409,
            detail=f"Outcome '{request.outcome}' was not trained in this artifact.",
        )

    bundle = joblib.load(artifact / "future_risk_models.joblib")[request.outcome]
    columns = json.loads((artifact / "feature_columns.json").read_text())

    visits = sorted(request.patient_history, key=lambda item: -float(item.get("days_before_index", 0)))
    frame_rows = []
    for index, visit in enumerate(visits):
        row = {"visit_index": index, "relative_time_days": -abs(float(visit.get("days_before_index", 0)))}
        for feature in PREVENTION_SAFE_FEATURES:
            value = visit.get(feature)
            row[f"feature_{feature}"] = None if value is None else float(value)
            row[f"mask_{feature}"] = int(value is not None)
        frame_rows.append(row)
    visit_frame = pd.DataFrame(frame_rows)
    visit_frame["delta_days_since_previous_visit"] = (
        visit_frame["relative_time_days"].diff().fillna(0.0).abs()
    )

    # Baseline tabular features, mirroring longitudinal_dataset.build_patient_features.
    feature_row: dict[str, Any] = {
        "visit_count": len(visit_frame),
        "visit_density_per_year": float(
            len(visit_frame) / max((-visit_frame["relative_time_days"].min()) / 365.25, 0.25)
        ),
        "history_days": float(-visit_frame["relative_time_days"].min()),
        "median_visit_gap_days": float(visit_frame["delta_days_since_previous_visit"].iloc[1:].median() or 0.0),
        "missingness_burden": float(
            1.0 - visit_frame[[f"mask_{f}" for f in PREVENTION_SAFE_FEATURES]].to_numpy().mean()
        ),
    }
    for feature in PREVENTION_SAFE_FEATURES:
        values = visit_frame[f"feature_{feature}"].astype(float)
        times = visit_frame["relative_time_days"].astype(float) / 365.25
        observed = values.notna()
        feature_row[f"{feature}_observed_count"] = int(observed.sum())
        if observed.sum() == 0:
            for suffix in ("last", "mean", "slope_per_year", "delta"):
                feature_row[f"{feature}_{suffix}"] = float("nan")
            continue
        feature_row[f"{feature}_last"] = float(values[observed].iloc[-1])
        feature_row[f"{feature}_mean"] = float(values[observed].mean())
        feature_row[f"{feature}_delta"] = float(values[observed].iloc[-1] - values[observed].iloc[0])
        feature_row[f"{feature}_slope_per_year"] = (
            float(np.polyfit(times[observed], values[observed], 1)[0])
            if observed.sum() >= 2 and times[observed].std() > 0
            else 0.0
        )
    design = pd.DataFrame([feature_row])
    for column in columns:
        if column not in design:
            design[column] = float("nan")
    matrix = design[columns].astype(float).to_numpy()

    horizons: dict[str, Any] = {}
    for horizon in HORIZON_DAYS:
        suffix = HORIZON_LABELS[horizon]
        baseline = bundle["baselines"].get(suffix, {}).get("models", {})
        selected = (metadata.get("selection") or {}).get(f"{request.outcome}:{suffix}", {})
        selected_model = selected.get("selected_model")
        entry: dict[str, Any] = {"selected_model": selected_model, "models": {}}
        for name, model in baseline.items():
            raw = float(model.predict_proba(matrix)[0, 1])
            calibrator = bundle["calibrators"].get(f"{name}:{suffix}")
            calibrated = raw
            if calibrator is not None:
                if calibrator["method"] == "isotonic":
                    calibrated = float(np.clip(calibrator["model"].predict([raw])[0], 0, 1))
                else:
                    calibrated = float(calibrator["model"].predict_proba([[raw]])[0, 1])
            entry["models"][name] = {
                "raw_cumulative_incidence": round(raw, 6),
                "calibrated_cumulative_incidence": round(calibrated, 6),
                "calibration_method": calibrator["method"] if calibrator else None,
            }
        if bundle.get("hazard", {}).get("status") == "fitted":
            from future_risk_models import hazard_cumulative_incidence

            hazard_value = float(hazard_cumulative_incidence(bundle["hazard"], design, horizon)[0])
            entry["models"]["discrete_time_hazard"] = {
                "raw_cumulative_incidence": round(hazard_value, 6),
                "calibrated_cumulative_incidence": None,
                "competing_event_adjusted": bundle["hazard"].get("competing_handled", False),
            }
        horizons[suffix] = entry

    return {
        "output_type": "simulated_cumulative_incidence",
        "simulation_only": True,
        "clinical_use": "prohibited",
        "explanation_class": "model_association",
        "causal_status": "causal_claim_not_established",
        "outcome": request.outcome,
        "artifact": {
            "dir": str(artifact),
            "capability_state": metadata["capability_state"],
            "code_version": metadata["code_version"],
            "created_at": metadata["created_at"],
        },
        "input_summary": {
            "visits": len(visit_frame),
            "history_days": feature_row["history_days"],
            "missingness_burden": round(feature_row["missingness_burden"], 4),
        },
        "horizons": horizons,
        "competing_outcomes_note": (
            "Cumulative incidence from the discrete-time model is cause-specific with death "
            "treated as a competing event. Site-specific cancer outputs are disabled."
        ),
        "calibration_state": (
            "Calibrated on a synthetic validation split. Absolute probabilities are not "
            "population-calibrated because the simulated cohort used declared enrichment strata."
        ),
        "banner": (
            "SIMULATION ONLY - synthetic data, software verification only. Not validated for "
            "patient risk, not a diagnosis, and not evidence of early detection."
        ),
    }
