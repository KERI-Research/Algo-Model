"""Model surface: metadata, single-record probe, and batch dataset scoring.

All scoring uses the deployed NumPy inference artifact
(``assets/ssl_artifact``): exported autoencoder weights, the fitted preprocessor
constants and the reference score distribution. No training and no PyTorch, in
any code path.

Scoring goes through :mod:`server.inference`, which replays the exported
preprocessor constants with NumPy alone (so serverless hosting does not need
scikit-learn or SciPy) and falls back to the vendored scikit-learn path when the
exported constants are absent. Outputs are identical either way, enforced by
``tests/test_inference_parity.py``.
"""

from __future__ import annotations

import csv
import io
import json
import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .core_bridge import CODE_VERSION, PREVENTION_FEATURES
from .inference import params_available, score_records

NON_DIAGNOSTIC_WARNING = (
    "Research use only, non-diagnostic. Outputs are metabolic deviation scores, "
    "reference percentiles and learned representations. They are not diagnoses, not "
    "cancer-type claims and not future disease probabilities. Clinician review is required."
)

FIELD_MEANINGS = {
    "metabolic_deviation_score": (
        "How unusual this metabolic profile is versus the training reference distribution. "
        "0.7 x robust reconstruction-error deviation + 0.3 x robust latent-distance deviation, "
        "floored at zero. Unitless and unbounded above."
    ),
    "reference_percentile": (
        "Rank of the deviation score inside the training reference distribution "
        "(63,041 scored adult NHANES records). 90 means more unusual than 90% of that "
        "reference sample. It is not a risk percentile."
    ),
    "top_deviation_features": (
        "Allowlisted features with the largest mean squared reconstruction error for this "
        "record. They indicate which measurements the model could not reproduce, not causes."
    ),
    "latent_representation": (
        "The learned 16-dimensional encoding of the record. Useful for similarity work; "
        "it carries no label."
    ),
}

SECTION_TITLES = {
    "current_profile_assessment": "Current profile assessment",
    "standout_factors": "Standout factors in this profile",
    "data_readiness": "Data readiness and missing information",
    "research_association": "Research-only cancer/diabetes association",
}

PROFILE_WARNING_STATEMENT = (
    "This profile differs from the reference and may warrant clinician review."
)

FUTURE_RISK_DISABLED_STATEMENT = (
    "No validated future cancer or diabetes risk model is currently deployed."
)


def _deviation_band_from_percentile(percentile: float) -> dict[str, str]:
    if percentile < 90:
        return {
            "key": "within_reference_range",
            "label": "Within reference range",
            "interpretation": "No model warning; not evidence disease is absent.",
        }
    if percentile < 95:
        return {
            "key": "mild_deviation",
            "label": "Mild deviation",
            "interpretation": "Consider data quality and routine review.",
        }
    if percentile < 99:
        return {
            "key": "elevated_deviation",
            "label": "Elevated deviation",
            "interpretation": "Clinician-reviewed follow-up research.",
        }
    return {
        "key": "high_deviation",
        "label": "High deviation",
        "interpretation": "Strongly unusual profile; still not diagnostic.",
    }

EVIDENCE_BOUNDARIES = [
    "Trained self-supervised on cross-sectional NHANES adult records; no outcome label was used.",
    "No patient-level longitudinal follow-up exists in the training data, so no time horizon "
    "(1, 3 or 5 year) can be estimated and the future-risk head stays fail-closed.",
    "A high percentile means the profile is unusual for the reference sample. It does not "
    "identify a disease, a cancer type or a site.",
    "The reference sample is US NHANES adults; profiles from other populations may sit at a "
    "different place in the distribution for reasons unrelated to health.",
    "Feature contributions are reconstruction diagnostics, not causal attributions.",
]


@lru_cache(maxsize=1)
def artifact_metadata() -> dict[str, Any]:
    return json.loads((config.SSL_ARTIFACT_DIR / "metadata.json").read_text())


def model_summary() -> dict[str, Any]:
    metadata = artifact_metadata()
    distribution = metadata.get("score_distribution", {})
    capabilities = metadata.get("capabilities", {})
    training_config = metadata.get("config", {})
    return {
        "model_name": metadata.get("model_name"),
        "model_version": config.MODEL_VERSION,
        "code_version": metadata.get("code_version", CODE_VERSION),
        "deployment_version": config.DEPLOYMENT_VERSION,
        "created_at": metadata.get("created_at"),
        "inference_backend": "numpy",
        "preprocessor_path": (
            "exported_constants" if params_available() else "sklearn_artifact"
        ),
        "training_backend_not_deployed": "PyTorch is not installed or used in this deployment.",
        "architecture": {
            "type": "Denoising autoencoder (self-supervised)",
            "input_features": len(metadata.get("features", [])),
            "transformed_dimension": metadata.get("transformed_dimension"),
            "latent_dimension": training_config.get("latent_dim", 16),
            "hidden_layers": training_config.get("hidden_dims"),
            "activation": "GELU",
            "objective": "Masked + noised input reconstruction (MSE)",
            "score_definition": (
                "0.7 x robust reconstruction deviation + 0.3 x robust latent deviation"
            ),
            "reference_rows": distribution.get("combined_sorted_note"),
        },
        "training_rows": metadata.get("training_rows"),
        "reference_rows_scored": metadata.get("adult_rows_scored"),
        "features": metadata.get("features", PREVENTION_FEATURES),
        "intended_use": metadata.get("intended_use"),
        "not_intended_for": metadata.get("not_intended_for", []),
        "capabilities": capabilities,
        "supported_outputs": [
            "metabolic_deviation_score",
            "reference_percentile",
            "latent_representation",
            "top_deviation_features",
        ],
        "prohibited_outputs": [
            "disease diagnosis or probability",
            "cancer type or site claim",
            "future-risk horizon (1/3/5 year) estimate",
            "clinical triage or treatment recommendation",
            "cluster labelled as a disease subtype",
        ],
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
    }


def score_single_record(record: dict[str, Any]) -> dict[str, Any]:
    """Score one explicitly submitted record. Returns research outputs only."""
    cleaned = {
        key: value
        for key, value in record.items()
        if key in PREVENTION_FEATURES and value not in (None, "")
    }
    if not cleaned:
        raise ValueError("No allowlisted feature values were supplied.")
    frame = pd.DataFrame([cleaned])
    result = score_records(frame, config.SSL_ARTIFACT_DIR)[0]
    deviation_band = _deviation_band_from_percentile(
        float(result.get("reference_percentile", 0.0))
    )
    features_missing = [
        feature for feature in PREVENTION_FEATURES if feature not in cleaned
    ]
    metadata = artifact_metadata()
    return {
        "score": result,
        "features_used": sorted(cleaned),
        "features_missing": features_missing,
        "field_meanings": FIELD_MEANINGS,
        "evidence_boundaries": EVIDENCE_BOUNDARIES,
        "dataset_capability_state": "Cross-sectional only",
        "patient_assessment": {
            "current_profile_assessment": {
                "section_title": SECTION_TITLES["current_profile_assessment"],
                "deviation_band": deviation_band["key"],
                "deviation_band_label": deviation_band["label"],
                "reference_percentile": result.get("reference_percentile"),
                "warning_label": f"{deviation_band['label']} (not diagnostic)",
                "note": PROFILE_WARNING_STATEMENT,
            },
            "standout_factors": {
                "section_title": SECTION_TITLES["standout_factors"],
                "top_deviation_features": result.get("top_deviation_features", []),
                "note": "Model association only; not causality and not diagnosis.",
            },
            "data_readiness": {
                "section_title": SECTION_TITLES["data_readiness"],
                "missing_fields": [
                    {
                        "field": field,
                        "priority": "medium",
                        "why_it_matters": "Additional features improve interpretation depth.",
                        "expected_impact_bucket": "moderate_confidence_gain",
                    }
                    for field in features_missing[:8]
                ],
                "dataset_capability_state": "Cross-sectional only",
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
        "output_type": "metabolic_deviation_and_representation",
        "is_future_risk_probability": False,
        "is_disease_classification": False,
        "artifact": {
            "model_name": metadata.get("model_name"),
            "code_version": metadata.get("code_version", CODE_VERSION),
            "created_at": metadata.get("created_at"),
            "backend": "numpy",
        },
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
    }


def _percentile_bands(percentiles: np.ndarray) -> dict[str, int]:
    bands = {
        "p0_to_p50": int(((percentiles >= 0) & (percentiles < 50)).sum()),
        "p50_to_p75": int(((percentiles >= 50) & (percentiles < 75)).sum()),
        "p75_to_p90": int(((percentiles >= 75) & (percentiles < 90)).sum()),
        "p90_to_p99": int(((percentiles >= 90) & (percentiles < 99)).sum()),
        "p99_and_above": int((percentiles >= 99).sum()),
    }
    return bands


def score_dataset(
    frame: pd.DataFrame,
    mapped_features: list[str],
    row_mask: pd.Series,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Score the accepted rows of an uploaded dataset. Capped and time-bounded."""
    started_at = started_at if started_at is not None else time.monotonic()
    accepted = frame.loc[row_mask]
    capped = accepted.head(config.MAX_SCORED_ROWS)
    if capped.empty:
        raise ValueError("No row in this file met the minimum feature requirement.")
    payload = capped[mapped_features].copy()
    payload.index = range(len(payload))
    results = score_records(payload, config.SSL_ARTIFACT_DIR)
    elapsed = time.monotonic() - started_at

    scores = np.array([item["metabolic_deviation_score"] for item in results], dtype=float)
    percentiles = np.array([item["reference_percentile"] for item in results], dtype=float)

    contribution_counts: dict[str, int] = {}
    for item in results:
        for entry in item["top_deviation_features"][:3]:
            contribution_counts[entry["feature"]] = (
                contribution_counts.get(entry["feature"], 0) + 1
            )
    ranked_contributions = sorted(
        (
            {
                "feature": feature,
                "rows_in_top_three": count,
                "share_of_rows": round(count / len(results), 4),
            }
            for feature, count in contribution_counts.items()
        ),
        key=lambda item: item["rows_in_top_three"],
        reverse=True,
    )[:10]

    rows: list[dict[str, Any]] = []
    row_ids = list(capped.index)
    for position, (row_id, item) in enumerate(zip(row_ids, results)):
        rows.append(
            {
                "row_index": int(row_id),
                "row_number": position + 1,
                "metabolic_deviation_score": item["metabolic_deviation_score"],
                "reference_percentile": item["reference_percentile"],
                "top_deviation_features": [
                    {
                        "feature": entry["feature"],
                        "reconstruction_error": round(entry["reconstruction_error"], 6),
                    }
                    for entry in item["top_deviation_features"][:3]
                ],
            }
        )

    return {
        "aggregate": {
            "rows_scored": len(results),
            "rows_accepted": int(row_mask.sum()),
            "rows_rejected": int(len(frame) - int(row_mask.sum())),
            "row_cap_applied": int(row_mask.sum()) > config.MAX_SCORED_ROWS,
            "row_cap": config.MAX_SCORED_ROWS,
            "compute_seconds": round(elapsed, 3),
            "deviation_score": {
                "min": round(float(scores.min()), 6),
                "median": round(float(np.median(scores)), 6),
                "mean": round(float(scores.mean()), 6),
                "p90": round(float(np.percentile(scores, 90)), 6),
                "max": round(float(scores.max()), 6),
            },
            "reference_percentile": {
                "median": round(float(np.median(percentiles)), 2),
                "mean": round(float(percentiles.mean()), 2),
                "share_at_or_above_p90": round(float((percentiles >= 90).mean()), 4),
            },
            "percentile_bands": _percentile_bands(percentiles),
            "top_deviation_features": ranked_contributions,
            "features_used": mapped_features,
        },
        "rows": rows,
        "field_meanings": FIELD_MEANINGS,
        "evidence_boundaries": EVIDENCE_BOUNDARIES,
        "interpretation": (
            "These are deviation statistics for the uploaded sample against the NHANES adult "
            "reference distribution. They are not prevalence, incidence or risk estimates, and "
            "no cluster or subgroup here may be described as a disease or cancer type."
        ),
        "clustering": {
            "available": False,
            "status": "unavailable_in_deployment",
            "reason": (
                "The validated clustering pipeline requires bootstrap and seed stability runs "
                "plus survey-cycle negative controls that exceed this deployment's compute "
                "budget. No cluster solution is produced for uploaded data."
            ),
        },
        "persistence": "none: the uploaded file was parsed in memory and discarded.",
        "non_diagnostic_warning": NON_DIAGNOSTIC_WARNING,
    }


def results_csv(rows: list[dict[str, Any]]) -> str:
    """Build the downloadable per-row results CSV entirely in memory."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "row_number",
            "source_row_index",
            "metabolic_deviation_score",
            "reference_percentile",
            "top_deviation_feature_1",
            "top_deviation_feature_2",
            "top_deviation_feature_3",
            "output_type",
            "is_diagnosis",
        ]
    )
    for row in rows:
        features = [entry["feature"] for entry in row.get("top_deviation_features", [])]
        features += [""] * (3 - len(features))
        writer.writerow(
            [
                row["row_number"],
                row["row_index"],
                row["metabolic_deviation_score"],
                row["reference_percentile"],
                *features[:3],
                "metabolic_deviation_and_representation",
                "no",
            ]
        )
    return buffer.getvalue()
