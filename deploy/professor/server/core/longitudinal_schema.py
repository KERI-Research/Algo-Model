"""Versioned longitudinal schema, contracts and capability gates for MetaboGuard.

This module defines the *only* accepted shape for time-stamped patient data, plus the
capability state machine that decides what such data may be used for.

Two schemas:

* **patient-event** (long format): one row per observation or outcome event.
* **patient-visit matrix** (wide format): one row per patient/visit, ready for temporal
  modelling, carrying relative time, visit density, time deltas and missingness masks.

Fail-closed principle: a dataset whose capability is ``simulation_only_longitudinal`` can
train and evaluate software, and can never unlock clinical or patient-facing risk output.
``assert_clinical_future_risk_allowed`` is the single chokepoint and currently raises for
every dataset this repository can produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "metaboguard-longitudinal-v1"
VISIT_MATRIX_VERSION = "metaboguard-visit-matrix-v1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Capability state machine
# ---------------------------------------------------------------------------


class CapabilityState(str, Enum):
    """What a dataset can support. Ordered from least to most capable."""

    CROSS_SECTIONAL = "cross_sectional"
    REPEATED_WITHOUT_OUTCOMES = "repeated_without_outcomes"
    SIMULATION_ONLY_LONGITUDINAL = "simulation_only_longitudinal"
    LONGITUDINAL_WITH_INCIDENT_OUTCOMES = "longitudinal_with_incident_outcomes"
    POST_DIAGNOSIS = "post_diagnosis"


#: What each state permits. `clinical_future_risk` is True for exactly one state, and no
#: dataset in this repository currently reaches it.
CAPABILITY_PERMISSIONS: dict[CapabilityState, dict[str, Any]] = {
    CapabilityState.CROSS_SECTIONAL: {
        "deviation_scoring": True,
        "representation_learning": True,
        "cross_sectional_association": True,
        "phenotype_clustering": True,
        "simulated_future_risk": False,
        "clinical_future_risk": False,
        "reason": "One observation per participant, no follow-up time.",
    },
    CapabilityState.REPEATED_WITHOUT_OUTCOMES: {
        "deviation_scoring": True,
        "representation_learning": True,
        "cross_sectional_association": True,
        "phenotype_clustering": True,
        "simulated_future_risk": False,
        "clinical_future_risk": False,
        "reason": "Repeated measurements exist but no incident outcome with an event date.",
    },
    CapabilityState.SIMULATION_ONLY_LONGITUDINAL: {
        "deviation_scoring": True,
        "representation_learning": True,
        "cross_sectional_association": True,
        "phenotype_clustering": True,
        "simulated_future_risk": True,
        "clinical_future_risk": False,
        "reason": (
            "Synthetic or simulated timelines. Valid for software and model engineering "
            "only. Metrics do not transfer to real patients and can never unlock clinical "
            "risk output."
        ),
    },
    CapabilityState.LONGITUDINAL_WITH_INCIDENT_OUTCOMES: {
        "deviation_scoring": True,
        "representation_learning": True,
        "cross_sectional_association": True,
        "phenotype_clustering": True,
        "simulated_future_risk": True,
        "clinical_future_risk": True,
        "reason": (
            "Real patient-level follow-up with incident outcomes and censoring. Still "
            "requires event-count gates, calibration review, clinician sign-off and "
            "external validation before any clinical statement."
        ),
    },
    CapabilityState.POST_DIAGNOSIS: {
        "deviation_scoring": False,
        "representation_learning": False,
        "cross_sectional_association": False,
        "phenotype_clustering": False,
        "simulated_future_risk": False,
        "clinical_future_risk": False,
        "reason": "Observed after diagnosis (for example TCGA). Cannot inform prevention.",
    },
}

#: Default intended horizons and the minimum event gate per horizon.
HORIZON_DAYS: tuple[int, ...] = (365, 1095, 1825)
HORIZON_LABELS: dict[int, str] = {365: "1y", 1095: "3y", 1825: "5y"}
MIN_EVENTS_PER_HORIZON = 50
MIN_NON_EVENTS_PER_HORIZON = 50

#: Outcomes this pipeline may model, and those permanently disabled.
SUPPORTED_OUTCOMES = ("type2_diabetes", "pan_cancer")
DISABLED_OUTCOMES: dict[str, str] = {
    "type1_diabetes": (
        "Research-only: requires autoantibodies, appropriate C-peptide and approved "
        "genetics. Never enabled from questionnaire or simulated proxies."
    ),
    "pancreatic_cancer": (
        "Site-specific pancreatic head stays disabled until a real cohort passes the "
        "50-event gate for the site. Historical supervised artifacts remain invalidated."
    ),
}
SITE_SPECIFIC_OUTCOME_PREFIX = "cancer_site_"


def capability_permissions(state: CapabilityState | str) -> dict[str, Any]:
    return CAPABILITY_PERMISSIONS[CapabilityState(state)]


def assert_capability_allows(state: CapabilityState | str, action: str) -> None:
    """Fail closed unless the dataset's capability state permits ``action``."""
    permissions = capability_permissions(state)
    if action not in permissions:
        raise KeyError(f"Unknown capability action '{action}'.")
    if not permissions[action]:
        raise PermissionError(
            f"Capability '{CapabilityState(state).value}' does not permit '{action}': "
            f"{permissions['reason']}"
        )


def assert_clinical_future_risk_allowed(state: CapabilityState | str) -> None:
    """The single chokepoint for real-patient future-risk output."""
    assert_capability_allows(state, "clinical_future_risk")


def assert_simulated_future_risk_allowed(
    state: CapabilityState | str, simulation_mode: bool
) -> None:
    """Simulated risk requires both a permitting capability and an explicit opt-in."""
    assert_capability_allows(state, "simulated_future_risk")
    if not simulation_mode:
        raise PermissionError(
            "Simulated future-risk scoring requires simulation_mode=true in the request. "
            "Refusing to return simulated risk that could be mistaken for clinical risk."
        )


def assert_outcome_allowed(outcome: str) -> None:
    if outcome in DISABLED_OUTCOMES:
        raise PermissionError(f"Outcome '{outcome}' is disabled: {DISABLED_OUTCOMES[outcome]}")
    if outcome.startswith(SITE_SPECIFIC_OUTCOME_PREFIX):
        return  # gated separately by event counts
    if outcome not in SUPPORTED_OUTCOMES:
        raise PermissionError(f"Outcome '{outcome}' is not a supported head.")


# ---------------------------------------------------------------------------
# Vocabularies, units and plausibility
# ---------------------------------------------------------------------------

EVENT_SOURCES = (
    "synthea",
    "metaboguard_simulator",
    "nhanes",
    "ehr",
    "registry",
    "manual",
)

MISSINGNESS_REASONS = (
    "observed",
    "not_measured_at_visit",
    "not_in_source_panel",
    "before_first_visit",
    "after_censoring",
    "invalid_value_removed",
    "unknown",
)

OUTCOME_TYPES = (
    "none",
    "type2_diabetes",
    "pan_cancer",
    "death",
    "censored",
)

#: Canonical unit per feature, with accepted alternates and conversions to canonical.
UNIT_HARMONISATION: dict[str, dict[str, Any]] = {
    "GLU_LBXGLU": {
        "canonical": "mg/dL",
        "conversions": {"mg/dl": 1.0, "mg/dL": 1.0, "mmol/l": 18.0182, "mmol/L": 18.0182},
    },
    "GHB_LBXGH": {
        "canonical": "%",
        # IFCC mmol/mol -> DCCT % : (mmol/mol * 0.09148) + 2.152
        "conversions": {"%": 1.0, "percent": 1.0},
        "affine": {"mmol/mol": (0.09148, 2.152)},
    },
    "TCHOL_LBXTC": {
        "canonical": "mg/dL",
        "conversions": {"mg/dl": 1.0, "mg/dL": 1.0, "mmol/l": 38.67, "mmol/L": 38.67},
    },
    "HDL_LBDHDD": {
        "canonical": "mg/dL",
        "conversions": {"mg/dl": 1.0, "mg/dL": 1.0, "mmol/l": 38.67, "mmol/L": 38.67},
    },
    "TRIGLY_LBXTR": {
        "canonical": "mg/dL",
        "conversions": {"mg/dl": 1.0, "mg/dL": 1.0, "mmol/l": 88.57, "mmol/L": 88.57},
    },
    "BMX_BMXBMI": {"canonical": "kg/m2", "conversions": {"kg/m2": 1.0}},
    "BMX_BMXWAIST": {
        "canonical": "cm",
        "conversions": {"cm": 1.0, "in": 2.54, "inch": 2.54},
    },
    "BMX_BMXWT": {"canonical": "kg", "conversions": {"kg": 1.0, "lb": 0.45359237}},
    "BPX_SYSTOLIC": {"canonical": "mmHg", "conversions": {"mmhg": 1.0, "mmHg": 1.0}},
    "INS_LBXIN": {"canonical": "uU/mL", "conversions": {"uu/ml": 1.0, "uU/mL": 1.0}},
    "DEMO_RIDAGEYR": {"canonical": "years", "conversions": {"years": 1.0, "year": 1.0}},
}

#: Wide review windows for impossible-value detection (not clinical reference intervals).
PLAUSIBLE_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "DEMO_RIDAGEYR": (0, 120),
    "BMX_BMXBMI": (8, 100),
    "BMX_BMXWAIST": (30, 220),
    "BMX_BMXWT": (2, 400),
    "GHB_LBXGH": (2.5, 20),
    "GLU_LBXGLU": (20, 800),
    "INS_LBXIN": (0, 400),
    "TCHOL_LBXTC": (50, 600),
    "HDL_LBDHDD": (5, 200),
    "TRIGLY_LBXTR": (10, 3000),
    "BPX_SYSTOLIC": (50, 260),
}

#: Prevention-safe longitudinal features. Outcome-derived columns are never inputs.
PREVENTION_SAFE_FEATURES = (
    "DEMO_RIDAGEYR",
    "BMX_BMXBMI",
    "BMX_BMXWAIST",
    "BMX_BMXWT",
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "INS_LBXIN",
    "TCHOL_LBXTC",
    "HDL_LBDHDD",
    "TRIGLY_LBXTR",
    "BPX_SYSTOLIC",
)

DENYLISTED_FEATURE_TOKENS = (
    "cancer",
    "diabetes_diagnosis",
    "outcome",
    "event_",
    "tcga_",
    "stage",
    "died",
    "death",
)


# ---------------------------------------------------------------------------
# Pydantic contracts
# ---------------------------------------------------------------------------


class PatientEvent(BaseModel):
    """One row of the patient-event (long) schema."""

    schema_version: str = SCHEMA_VERSION
    patient_id: str
    observation_timestamp: datetime
    source: str
    feature_code: str
    value: float | None = None
    unit: str | None = None
    missingness_reason: str = "observed"
    visit_id: str | None = None
    index_date: datetime | None = None
    outcome_type: str = "none"
    event_date: datetime | None = None
    cancer_site: str | None = None
    cancer_stage: str | None = None
    censoring_date: datetime | None = None
    provenance: str = Field(..., description="Generator/version/seed or source file.")

    @field_validator("source")
    @classmethod
    def _source_known(cls, value: str) -> str:
        if value not in EVENT_SOURCES:
            raise ValueError(f"source must be one of {EVENT_SOURCES}")
        return value

    @field_validator("missingness_reason")
    @classmethod
    def _missingness_known(cls, value: str) -> str:
        if value not in MISSINGNESS_REASONS:
            raise ValueError(f"missingness_reason must be one of {MISSINGNESS_REASONS}")
        return value

    @field_validator("outcome_type")
    @classmethod
    def _outcome_known(cls, value: str) -> str:
        if value not in OUTCOME_TYPES and not value.startswith(SITE_SPECIFIC_OUTCOME_PREFIX):
            raise ValueError(f"outcome_type must be one of {OUTCOME_TYPES} or a site outcome")
        return value

    @model_validator(mode="after")
    def _consistency(self) -> "PatientEvent":
        if self.value is None and self.missingness_reason == "observed":
            raise ValueError("A missing value must state a missingness_reason other than 'observed'.")
        if self.outcome_type not in {"none", "censored"} and self.event_date is None:
            raise ValueError("An outcome row requires an event_date.")
        if self.event_date and self.index_date and self.event_date < self.index_date:
            raise ValueError("event_date precedes index_date.")
        if self.censoring_date and self.index_date and self.censoring_date < self.index_date:
            raise ValueError("censoring_date precedes index_date.")
        if self.cancer_site and not (
            self.outcome_type == "pan_cancer"
            or self.outcome_type.startswith(SITE_SPECIFIC_OUTCOME_PREFIX)
        ):
            raise ValueError("cancer_site is only valid on a cancer outcome row.")
        return self


class VisitRow(BaseModel):
    """One row of the patient-visit matrix schema."""

    schema_version: str = VISIT_MATRIX_VERSION
    patient_id: str
    visit_id: str
    visit_index: int
    visit_timestamp: datetime
    index_date: datetime
    relative_time_days: float
    delta_days_since_previous_visit: float
    visit_density_per_year: float
    observed_feature_count: int
    features: dict[str, float | None]
    feature_masks: dict[str, int]

    @model_validator(mode="after")
    def _feature_safety(self) -> "VisitRow":
        for name in self.features:
            lowered = name.lower()
            if any(token in lowered for token in DENYLISTED_FEATURE_TOKENS):
                raise ValueError(f"Feature '{name}' is denylisted as a model input.")
        if set(self.features) != set(self.feature_masks):
            raise ValueError("features and feature_masks must cover the same keys.")
        if self.relative_time_days > 0:
            raise ValueError(
                "Visit matrix rows must be at or before the index date "
                "(relative_time_days <= 0); later visits leak future information."
            )
        return self


class DatasetManifest(BaseModel):
    """Versioned manifest written beside every generated dataset."""

    schema_version: str = SCHEMA_VERSION
    dataset_name: str
    created_at: datetime
    capability_state: CapabilityState
    simulation_only: bool
    generator: dict[str, Any]
    row_counts: dict[str, int]
    fingerprints: dict[str, str]
    horizons_days: list[int] = Field(default_factory=lambda: list(HORIZON_DAYS))
    outcomes: list[str] = Field(default_factory=lambda: list(SUPPORTED_OUTCOMES))
    disabled_outcomes: dict[str, str] = Field(default_factory=lambda: dict(DISABLED_OUTCOMES))
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _simulation_consistency(self) -> "DatasetManifest":
        if self.capability_state == CapabilityState.SIMULATION_ONLY_LONGITUDINAL and not self.simulation_only:
            raise ValueError("simulation_only must be true for a simulation-only capability.")
        if self.simulation_only and self.capability_state == CapabilityState.LONGITUDINAL_WITH_INCIDENT_OUTCOMES:
            raise ValueError("Simulated data may not claim real longitudinal capability.")
        return self


def json_schemas() -> dict[str, Any]:
    """JSON Schema documents for the three contracts (for review and external tooling)."""
    return {
        "patient_event": PatientEvent.model_json_schema(),
        "visit_row": VisitRow.model_json_schema(),
        "dataset_manifest": DatasetManifest.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# Validation of a long-format frame
# ---------------------------------------------------------------------------

EVENT_COLUMNS = tuple(PatientEvent.model_fields.keys())


@dataclass
class ValidationIssue:
    code: str
    level: str  # "hard" | "soft"
    message: str
    rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "level": self.level, "message": self.message, "rows": self.rows}


@dataclass
class EventValidationReport:
    dataset: str
    generated_at: str
    row_counts: dict[str, int] = field(default_factory=dict)
    unit_harmonisation: dict[str, Any] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    ordering: dict[str, Any] = field(default_factory=dict)
    impossible_values: dict[str, Any] = field(default_factory=dict)
    missingness: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def hard_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "hard"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_type": "longitudinal_event_validation",
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "status": "blocked" if self.hard_issues else "ok",
            "row_counts": self.row_counts,
            "unit_harmonisation": self.unit_harmonisation,
            "duplicates": self.duplicates,
            "timestamp_ordering": self.ordering,
            "impossible_values": self.impossible_values,
            "missingness": self.missingness,
            "outcomes": self.outcomes,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def harmonise_units(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert every value to its feature's canonical unit; report what was converted."""
    result = frame.copy()
    log: dict[str, Any] = {"converted": {}, "unknown_units": {}, "canonical_units": {}}
    for feature, spec in UNIT_HARMONISATION.items():
        mask = result["feature_code"] == feature
        if not mask.any():
            continue
        log["canonical_units"][feature] = spec["canonical"]
        units = result.loc[mask, "unit"].fillna(spec["canonical"])
        for unit_value in sorted(set(units.astype(str))):
            if unit_value == spec["canonical"]:
                continue
            selector = mask & (result["unit"].astype(str) == unit_value)
            factor = spec.get("conversions", {}).get(unit_value)
            affine = spec.get("affine", {}).get(unit_value)
            if factor is not None:
                result.loc[selector, "value"] = result.loc[selector, "value"] * factor
            elif affine is not None:
                slope, intercept = affine
                result.loc[selector, "value"] = (
                    result.loc[selector, "value"] * slope + intercept
                )
            else:
                log["unknown_units"].setdefault(feature, []).append(unit_value)
                continue
            result.loc[selector, "unit"] = spec["canonical"]
            log["converted"].setdefault(feature, {})[unit_value] = int(selector.sum())
    return result, log


def validate_event_frame(
    frame: pd.DataFrame, dataset_name: str = "unnamed", strict: bool = True
) -> tuple[pd.DataFrame, EventValidationReport]:
    """Validate, harmonise and clean a patient-event frame. Fail-closed when strict."""
    report = EventValidationReport(dataset=dataset_name, generated_at=datetime.now(UTC).isoformat())

    missing_columns = [column for column in EVENT_COLUMNS if column not in frame.columns]
    if missing_columns:
        report.issues.append(
            ValidationIssue("missing_columns", "hard", f"Missing schema columns: {missing_columns}")
        )
        if strict:
            raise ValueError(f"Missing schema columns: {missing_columns}")
        return frame, report

    working = frame.copy()
    # format="ISO8601" so timestamps with and without fractional seconds both parse.
    working["observation_timestamp"] = pd.to_datetime(
        working["observation_timestamp"], utc=True, errors="coerce", format="ISO8601"
    )
    for column in ("index_date", "event_date", "censoring_date"):
        working[column] = pd.to_datetime(
            working[column], utc=True, errors="coerce", format="ISO8601"
        )
    working["value"] = pd.to_numeric(working["value"], errors="coerce")

    report.row_counts = {
        "rows": int(len(working)),
        "patients": int(working["patient_id"].nunique()),
        "visits": int(working["visit_id"].nunique(dropna=True)),
        "observation_rows": int((working["outcome_type"] == "none").sum()),
        "outcome_rows": int((working["outcome_type"] != "none").sum()),
    }

    bad_timestamps = int(working["observation_timestamp"].isna().sum())
    if bad_timestamps:
        report.issues.append(
            ValidationIssue("unparseable_timestamp", "hard", "Unparseable observation timestamps.", bad_timestamps)
        )

    unknown_reasons = sorted(set(working["missingness_reason"].dropna()) - set(MISSINGNESS_REASONS))
    if unknown_reasons:
        report.issues.append(
            ValidationIssue("unknown_missingness_reason", "hard", f"Unknown reasons: {unknown_reasons}")
        )
    unknown_outcomes = sorted(
        value
        for value in set(working["outcome_type"].dropna())
        if value not in OUTCOME_TYPES and not str(value).startswith(SITE_SPECIFIC_OUTCOME_PREFIX)
    )
    if unknown_outcomes:
        report.issues.append(
            ValidationIssue("unknown_outcome_type", "hard", f"Unknown outcome types: {unknown_outcomes}")
        )
    if working["provenance"].isna().any():
        report.issues.append(
            ValidationIssue("missing_provenance", "hard", "Every row needs provenance.", int(working["provenance"].isna().sum()))
        )

    working, unit_log = harmonise_units(working)
    report.unit_harmonisation = unit_log
    if unit_log["unknown_units"]:
        report.issues.append(
            ValidationIssue("unknown_unit", "hard", f"Unconvertible units: {unit_log['unknown_units']}")
        )

    # Duplicate resolution: same patient/timestamp/feature keeps the last row, which is the
    # most recently appended provenance. The count is reported, never silently dropped.
    key = ["patient_id", "observation_timestamp", "feature_code", "outcome_type"]
    duplicated = working.duplicated(subset=key, keep="last")
    report.duplicates = {
        "duplicate_rows_removed": int(duplicated.sum()),
        "resolution_rule": "keep last row per (patient_id, observation_timestamp, feature_code, outcome_type)",
    }
    working = working[~duplicated].copy()

    working = working.sort_values(["patient_id", "observation_timestamp"], kind="stable").reset_index(drop=True)
    out_of_order = 0
    for _, group in working.groupby("patient_id", sort=False):
        stamps = group["observation_timestamp"].to_numpy()
        if len(stamps) > 1:
            gaps = np.diff(stamps) / np.timedelta64(1, "s")
            out_of_order += int((gaps < 0).sum())
    report.ordering = {
        "sorted_by": "patient_id, observation_timestamp",
        "out_of_order_pairs_after_sort": out_of_order,
    }
    if out_of_order:
        report.issues.append(
            ValidationIssue("timestamp_ordering", "hard", "Timestamps remain out of order after sorting.", out_of_order)
        )

    impossible: dict[str, Any] = {}
    for feature, (low, high) in PLAUSIBLE_VALUE_RANGES.items():
        mask = working["feature_code"] == feature
        if not mask.any():
            continue
        values = working.loc[mask, "value"]
        bad = mask & ((working["value"] < low) | (working["value"] > high))
        count = int(bad.sum())
        if count:
            working.loc[bad, "value"] = np.nan
            working.loc[bad, "missingness_reason"] = "invalid_value_removed"
        impossible[feature] = {
            "window": [low, high],
            "observed": int(values.notna().sum()),
            "removed": count,
        }
        if values.notna().sum() and count / max(values.notna().sum(), 1) > 0.10:
            report.issues.append(
                ValidationIssue(
                    "impossible_value_burden",
                    "hard",
                    f"{feature}: more than 10% of values fell outside {low}-{high}.",
                    count,
                )
            )
    report.impossible_values = impossible

    observation_rows = working[working["outcome_type"] == "none"]
    report.missingness = {
        "rows_with_missing_value": int(observation_rows["value"].isna().sum()),
        "by_reason": {
            reason: int(count)
            for reason, count in observation_rows["missingness_reason"].value_counts().items()
        },
        "per_feature_coverage": {
            feature: round(
                float(
                    observation_rows.loc[observation_rows["feature_code"] == feature, "value"].notna().mean()
                ),
                6,
            )
            for feature in sorted(observation_rows["feature_code"].dropna().unique())
        },
    }

    outcome_rows = working[working["outcome_type"] != "none"]
    report.outcomes = {
        "counts": {
            str(name): int(count) for name, count in outcome_rows["outcome_type"].value_counts().items()
        },
        "patients_with_any_outcome": int(outcome_rows["patient_id"].nunique()),
        "cancer_sites": {
            str(name): int(count)
            for name, count in outcome_rows["cancer_site"].dropna().value_counts().items()
        },
    }

    if strict and report.hard_issues:
        raise ValueError(
            "Longitudinal schema validation failed: "
            + "; ".join(issue.message for issue in report.hard_issues)
        )
    return working, report


# ---------------------------------------------------------------------------
# Visit matrix construction
# ---------------------------------------------------------------------------


def build_visit_matrix(
    events: pd.DataFrame,
    features: tuple[str, ...] = PREVENTION_SAFE_FEATURES,
    max_visits: int | None = None,
) -> pd.DataFrame:
    """Build the patient-visit matrix from validated events.

    Only observations at or before each patient's index date are used, so nothing after the
    prediction time can enter a model input. Each row carries relative time, the gap since
    the previous visit, visit density, per-feature values and per-feature observed masks.
    """
    observations = events[events["outcome_type"] == "none"].copy()
    observations = observations[observations["feature_code"].isin(features)]
    index_dates = (
        events.groupby("patient_id")["index_date"].max().rename("index_date_resolved")
    )
    observations = observations.join(index_dates, on="patient_id")
    observations = observations[
        observations["observation_timestamp"] <= observations["index_date_resolved"]
    ]

    rows: list[dict[str, Any]] = []
    for patient_id, group in observations.groupby("patient_id", sort=True):
        index_date = group["index_date_resolved"].iloc[0]
        visits = sorted(group["observation_timestamp"].unique())
        if max_visits:
            visits = visits[-max_visits:]
        previous: Any = None
        span_days = max((visits[-1] - visits[0]) / np.timedelta64(1, "D"), 1.0) if visits else 1.0
        for visit_index, timestamp in enumerate(visits):
            slice_frame = group[group["observation_timestamp"] == timestamp]
            values = {
                feature: (
                    float(slice_frame.loc[slice_frame["feature_code"] == feature, "value"].iloc[0])
                    if (slice_frame["feature_code"] == feature).any()
                    and pd.notna(slice_frame.loc[slice_frame["feature_code"] == feature, "value"].iloc[0])
                    else None
                )
                for feature in features
            }
            row = {
                "schema_version": VISIT_MATRIX_VERSION,
                "patient_id": patient_id,
                "visit_id": f"{patient_id}:{visit_index}",
                "visit_index": visit_index,
                "visit_timestamp": pd.Timestamp(timestamp),
                "index_date": pd.Timestamp(index_date),
                "relative_time_days": float((timestamp - index_date) / np.timedelta64(1, "D")),
                "delta_days_since_previous_visit": (
                    0.0 if previous is None else float((timestamp - previous) / np.timedelta64(1, "D"))
                ),
                "visit_density_per_year": round(float(len(visits) / (span_days / 365.25)), 4),
                "observed_feature_count": int(sum(1 for value in values.values() if value is not None)),
            }
            for feature, value in values.items():
                row[f"feature_{feature}"] = value
                row[f"mask_{feature}"] = int(value is not None)
            rows.append(row)
            previous = timestamp
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        raise ValueError("Visit matrix is empty: no observations at or before the index dates.")
    return matrix.sort_values(["patient_id", "visit_index"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fingerprints and manifests
# ---------------------------------------------------------------------------


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Content fingerprint that is stable across row order but sensitive to values."""
    canonical = frame.sort_index(axis=1)
    payload = pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest.model_dump_json(indent=2))
    return output


def horizon_gate(
    labels: pd.DataFrame, outcome: str, horizons: tuple[int, ...] = HORIZON_DAYS
) -> dict[str, Any]:
    """Evaluate the 50-event / 50-non-event gate per horizon on a label frame."""
    per_horizon: dict[str, Any] = {}
    eligible_any = False
    for horizon in horizons:
        label_column = f"{outcome}_{HORIZON_LABELS[horizon]}_label"
        mask_column = f"{outcome}_{HORIZON_LABELS[horizon]}_eligible"
        if label_column not in labels.columns or mask_column not in labels.columns:
            per_horizon[HORIZON_LABELS[horizon]] = {
                "eligible": False,
                "events": 0,
                "non_events": 0,
                "reason": "labels not computed",
            }
            continue
        eligible = labels[mask_column] == 1
        events = int((labels.loc[eligible, label_column] == 1).sum())
        non_events = int((labels.loc[eligible, label_column] == 0).sum())
        passes = events >= MIN_EVENTS_PER_HORIZON and non_events >= MIN_NON_EVENTS_PER_HORIZON
        eligible_any = eligible_any or passes
        per_horizon[HORIZON_LABELS[horizon]] = {
            "eligible": passes,
            "events": events,
            "non_events": non_events,
            "eligible_patients": int(eligible.sum()),
            "reason": None if passes else f"below the {MIN_EVENTS_PER_HORIZON}/{MIN_NON_EVENTS_PER_HORIZON} gate",
        }
    return {
        "outcome": outcome,
        "minimum_events": MIN_EVENTS_PER_HORIZON,
        "minimum_non_events": MIN_NON_EVENTS_PER_HORIZON,
        "per_horizon": per_horizon,
        "any_horizon_eligible": eligible_any,
    }