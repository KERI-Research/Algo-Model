from __future__ import annotations

from typing import Any

RESEARCH_SYSTEM_LABEL = "research system for preventive monitoring"
NON_DIAGNOSTIC_WARNING = (
    "Research use only, non-diagnostic. Outputs are metabolic deviation scores, "
    "reference percentiles, latent representations, or cross-sectional associations "
    "with already-recorded diagnoses. They are not diagnoses and not future disease "
    "probabilities. Clinician review is required."
)
FUTURE_RISK_DISABLED_STATEMENT = (
    "MetaboGuard currently has no validated future cancer or diabetes risk model on "
    "this dataset capability state."
)
PROFILE_WARNING_STATEMENT = (
    "This profile differs from the reference and may warrant clinician review."
)

SECTION_TITLES = {
    "current_profile_assessment": "Current profile assessment",
    "standout_factors": "Standout factors in this profile",
    "data_readiness": "Data readiness and missing information",
    "research_association": "Research-only cancer/diabetes association",
}

DATASET_CAPABILITY_STATES = {
    "cross_sectional_representation_and_deviation_only": "Cross-sectional only",
    "multi_horizon_risk": "Longitudinal with incident outcomes",
}

DEVIATION_BANDS = [
    {
        "key": "within_reference_range",
        "label": "Within reference range",
        "minimum": 0.0,
        "maximum": 90.0,
        "interpretation": "No model warning; not evidence disease is absent.",
    },
    {
        "key": "mild_deviation",
        "label": "Mild deviation",
        "minimum": 90.0,
        "maximum": 95.0,
        "interpretation": "Consider data quality and routine review.",
    },
    {
        "key": "elevated_deviation",
        "label": "Elevated deviation",
        "minimum": 95.0,
        "maximum": 99.0,
        "interpretation": "Clinician-reviewed follow-up research.",
    },
    {
        "key": "high_deviation",
        "label": "High deviation",
        "minimum": 99.0,
        "maximum": 100.0,
        "interpretation": "Strongly unusual profile; still not diagnostic.",
    },
]

ASSOCIATION_SCOPE_NOTES = {
    "any_cancer_prevalence": {
        "status": "research_association_only",
        "note": "Cross-sectional association; not future cancer risk.",
    },
    "type2_diabetes_proxy": {
        "status": "current_state_association",
        "note": "Current-state association only; does not predict future development.",
    },
    "type1_proxy": {
        "status": "unvalidated_research_proxy",
        "note": "Unvalidated research proxy; never a diagnosis or development-risk output.",
    },
}


def capability_state_from_supported_output(supported_output: str | None) -> str:
    if supported_output in DATASET_CAPABILITY_STATES:
        return DATASET_CAPABILITY_STATES[supported_output]
    return "Cross-sectional only"


def deviation_band_from_percentile(percentile: float | int | None) -> dict[str, Any]:
    try:
        value = float(percentile)
    except (TypeError, ValueError):
        return {
            "key": "unavailable",
            "label": "Unavailable",
            "interpretation": "Reference percentile was not available for banding.",
        }

    clamped = max(0.0, min(100.0, value))
    for band in DEVIATION_BANDS:
        lower = band["minimum"]
        upper = band["maximum"]
        if band["key"] == "high_deviation":
            if clamped >= lower:
                return {
                    "key": band["key"],
                    "label": band["label"],
                    "interpretation": band["interpretation"],
                }
            continue
        if lower <= clamped < upper:
            return {
                "key": band["key"],
                "label": band["label"],
                "interpretation": band["interpretation"],
            }

    return {
        "key": "within_reference_range",
        "label": "Within reference range",
        "interpretation": "No model warning; not evidence disease is absent.",
    }
