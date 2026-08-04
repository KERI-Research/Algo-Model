"""Ephemeral CSV intake, screening and analysis for the professor dashboard.

Privacy posture
---------------
* Uploaded bytes are held in memory only. Nothing is written to disk, no
  temporary file is created, and no third party is contacted.
* Every code path that touches the upload buffer runs through
  :func:`shred_buffer` in a ``finally`` block, which overwrites the bytes before
  releasing them.
* Direct-identifier columns cause an outright rejection before any parsing of
  values is reported back to the client.
* Nothing about an upload survives the request: no cache, no database, no log
  line containing row values.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .core_bridge import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_EARLY_WARNING_FEATURES,
    MAX_IMPLAUSIBLE_FRACTION,
    MIN_COVERAGE_QUALIFIED,
    MIN_COVERAGE_USABLE,
    PLAUSIBLE_RANGES,
    PREVENTION_FEATURES,
    TIER_DEFINITIONS,
    dataset_capabilities,
    is_denylisted_input,
)

#: Column names that indicate a direct identifier. Matching is case-insensitive
#: and ignores spaces, hyphens and underscores.
DIRECT_IDENTIFIER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^(full)?name$", "personal name"),
    (r"^(first|last|fore|sur|given|family|middle|maiden)name$", "personal name"),
    (r"^(patient|participant|subject|person|staff|clinician|doctor|gp)name$", "personal name"),
    (r"^initials$", "personal name"),
    (r"e?mail(address)?$", "email address"),
    (r"^(tel|telephone|phone|mobile|cell|fax)(number|no)?$", "telephone number"),
    (r"^(contact|home|work)(tel|telephone|phone|number)$", "telephone number"),
    (r"^(address|address1|address2|addressline1|addressline2|streetaddress|street)$", "postal address"),
    (r"^(postcode|postalcode|zip|zipcode)$", "postcode or ZIP code"),
    (r"nhs(number|no|id)$", "NHS number"),
    (r"^(ssn|socialsecurity(number|no)?)$", "social security number"),
    (r"^(mrn|medicalrecord(number|no|id)?|hospital(number|no|id)|chinumber|chino)$",
     "medical record number"),
    (r"^(dob|dateofbirth|birthdate|birthday|datenaissance)$", "date of birth"),
    (r"^(passport|drivinglicence|driverslicense|insurancenumber|nino)$", "government identifier"),
    (r"^(ipaddress|deviceid|imei)$", "device identifier"),
    (r"^(gpspractice|gpname|nextofkin|emergencycontact)$", "contact detail"),
)

#: Value-level patterns. Sampled, never echoed back.
VALUE_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"), "email address"),
    (re.compile(r"^\d{3}-\d{2}-\d{4}$"), "social security number"),
    (re.compile(r"^\d{3}[ -]?\d{3}[ -]?\d{4}$"), "NHS number"),
    (re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", re.I), "UK postcode"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "exact date (possible date of birth)"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "exact date (possible date of birth)"),
)

#: Columns that are always allowed even though they look identifier-adjacent.
ALLOWED_ID_LIKE = {
    "id", "rowid", "row", "recordid", "seqn", "participantid", "patientid",
    "subjectid", "sampleid", "caseid", "anonid", "pseudoid", "studyid", "index",
}

ALLOWED_AGE_COLUMNS = {"age", "ageyears", "ageyrs", "demoridageyr"}


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def shred_buffer(buffer: bytearray | None) -> None:
    """Overwrite an upload buffer in place, then clear it."""
    if buffer is None:
        return
    try:
        for index in range(len(buffer)):
            buffer[index] = 0
    except (TypeError, IndexError):  # pragma: no cover - defensive
        pass
    finally:
        del buffer[:]


class DatasetRejected(Exception):
    """Raised when a dataset must not be processed at all."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class ColumnScreening:
    identifier_columns: list[dict[str, str]]
    prohibited_columns: list[dict[str, str]]
    mapped_features: list[str]
    unmapped_columns: list[str]
    missing_features: list[str]


def screen_column_names(columns: list[str]) -> list[dict[str, str]]:
    """Return direct-identifier findings for a list of column names."""
    findings: list[dict[str, str]] = []
    for column in columns:
        key = _normalise(column)
        if key in ALLOWED_ID_LIKE or key in ALLOWED_AGE_COLUMNS:
            continue
        for pattern, kind in DIRECT_IDENTIFIER_PATTERNS:
            if re.search(pattern, key):
                findings.append({"column": str(column), "identifier_type": kind})
                break
    return findings


def screen_column_values(frame: pd.DataFrame, sample_rows: int = 200) -> list[dict[str, str]]:
    """Sample object columns for direct-identifier value shapes."""
    findings: list[dict[str, str]] = []
    sample = frame.head(sample_rows)
    for column in sample.columns:
        series = sample[column]
        if series.dtype.kind not in {"O", "U", "S"}:
            continue
        values = [str(value).strip() for value in series.dropna().tolist()[:sample_rows]]
        if not values:
            continue
        for pattern, kind in VALUE_IDENTIFIER_PATTERNS:
            hits = sum(1 for value in values if pattern.match(value))
            if hits and hits / len(values) >= 0.2:
                findings.append({"column": str(column), "identifier_type": kind})
                break
    return findings


def prohibited_columns(columns: list[str]) -> list[dict[str, str]]:
    """Outcome, label-derived, post-diagnosis and TCGA leakage fields."""
    findings: list[dict[str, str]] = []
    for column in columns:
        name = str(column)
        if is_denylisted_input(name):
            reason = (
                "Post-diagnosis TCGA context column."
                if name.startswith("tcga_")
                else "Outcome label or label-derived column."
            )
            findings.append({"column": name, "reason": reason})
        elif name in FORBIDDEN_EARLY_WARNING_FEATURES:
            findings.append(
                {"column": name, "reason": "Prohibited as an early-warning model input."}
            )
    return findings


def read_csv_bytes(buffer: bytearray, filename: str) -> pd.DataFrame:
    """Parse an in-memory CSV upload with strict limits. Never touches disk."""
    if not filename.lower().endswith(".csv"):
        raise DatasetRejected("Only .csv files are accepted.")
    if len(buffer) == 0:
        raise DatasetRejected("The uploaded file is empty.")
    if len(buffer) > config.MAX_UPLOAD_BYTES:
        raise DatasetRejected(
            f"File exceeds the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )
    stream = io.BytesIO(bytes(buffer))
    try:
        frame = pd.read_csv(
            stream,
            low_memory=False,
            nrows=config.MAX_UPLOAD_ROWS + 1,
            encoding_errors="replace",
        )
    except UnicodeDecodeError:
        raise DatasetRejected("The file is not readable as UTF-8 text CSV.")
    except pd.errors.EmptyDataError:
        raise DatasetRejected("No CSV header row was found.")
    except pd.errors.ParserError:
        raise DatasetRejected("The file could not be parsed as CSV.")
    finally:
        stream.close()
    if frame.shape[0] > config.MAX_UPLOAD_ROWS:
        raise DatasetRejected(
            f"File exceeds the {config.MAX_UPLOAD_ROWS:,}-row limit for this deployment."
        )
    if frame.shape[1] == 0:
        raise DatasetRejected("No columns were found in the CSV header.")
    return frame


def screen_dataset(frame: pd.DataFrame) -> ColumnScreening:
    columns = [str(column) for column in frame.columns]
    identifiers = screen_column_names(columns) + [
        finding
        for finding in screen_column_values(frame)
        if finding["column"] not in {item["column"] for item in screen_column_names(columns)}
    ]
    if identifiers:
        raise DatasetRejected(
            "The file contains direct identifiers. De-identify it before upload.",
            {"identifier_columns": identifiers},
        )
    prohibited = prohibited_columns(columns)
    prohibited_names = {item["column"] for item in prohibited}
    mapped = [feature for feature in PREVENTION_FEATURES if feature in columns]
    return ColumnScreening(
        identifier_columns=[],
        prohibited_columns=prohibited,
        mapped_features=mapped,
        unmapped_columns=[
            column
            for column in columns
            if column not in PREVENTION_FEATURES and column not in prohibited_names
        ],
        missing_features=[feature for feature in PREVENTION_FEATURES if feature not in columns],
    )


def _coverage(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    total = max(len(frame), 1)
    return {
        feature: round(float(frame[feature].notna().sum()) / total, 6) for feature in features
    }


def _range_violations(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    violations: dict[str, Any] = {}
    for feature in features:
        bounds = PLAUSIBLE_RANGES.get(feature)
        if bounds is None:
            continue
        low, high = bounds
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        present = int(numeric.notna().sum())
        if present == 0:
            continue
        outside = int(((numeric < low) | (numeric > high)).sum())
        if outside:
            violations[feature] = {
                "plausible_range": [low, high],
                "values_outside_range": outside,
                "fraction_of_present_values": round(outside / present, 6),
                "exceeds_tolerance": (outside / present) > MAX_IMPLAUSIBLE_FRACTION,
            }
    return violations


def _eligibility(
    frame: pd.DataFrame,
    screening: ColumnScreening,
    coverage: dict[str, float],
    violations: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    eligibility: dict[str, dict[str, Any]] = {}
    for feature in PREVENTION_FEATURES:
        if feature not in screening.mapped_features:
            eligibility[feature] = {
                "tier": "unavailable",
                "reasons": ["Column absent from this file."],
                "coverage": 0.0,
            }
            continue
        column_coverage = coverage.get(feature, 0.0)
        reasons: list[str] = []
        tier = "usable_now"
        if column_coverage < MIN_COVERAGE_QUALIFIED:
            tier = "unavailable"
            reasons.append(
                f"Coverage {column_coverage * 100:.1f}% is below the "
                f"{MIN_COVERAGE_QUALIFIED * 100:.0f}% floor."
            )
        elif column_coverage < MIN_COVERAGE_USABLE:
            tier = "qualified_use"
            reasons.append(
                f"Coverage {column_coverage * 100:.1f}% is below the "
                f"{MIN_COVERAGE_USABLE * 100:.0f}% threshold."
            )
        if feature in violations:
            if violations[feature]["exceeds_tolerance"]:
                tier = "qualified_use" if tier == "usable_now" else tier
                reasons.append(
                    "Implausible-value burden "
                    f"{violations[feature]['fraction_of_present_values'] * 100:.2f}% exceeds the "
                    f"{MAX_IMPLAUSIBLE_FRACTION * 100:.0f}% tolerance."
                )
            else:
                reasons.append("A small number of values fall outside the plausible range.")
        if not reasons:
            reasons.append("Present, plausible and adequately covered.")
        eligibility[feature] = {
            "tier": tier,
            "reasons": reasons,
            "coverage": column_coverage,
        }
    for item in screening.prohibited_columns:
        eligibility[item["column"]] = {
            "tier": "prohibited",
            "reasons": [item["reason"]],
            "coverage": None,
        }
    return eligibility


def _preview(frame: pd.DataFrame, features: list[str], rows: int = 5) -> dict[str, Any]:
    """Preview only allowlisted feature columns, so nothing unexpected is echoed."""
    if not features:
        return {"columns": [], "rows": []}
    subset = frame[features].head(rows)
    records: list[list[Any]] = []
    for _, row in subset.iterrows():
        record = []
        for value in row.tolist():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                record.append(None)
            elif isinstance(value, (np.integer, np.floating)):
                record.append(float(value))
            else:
                record.append(str(value)[:32])
        records.append(record)
    return {"columns": features, "rows": records}


def eligible_row_mask(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    """A row is scoreable when age and at least three mapped features are present."""
    if not features:
        return pd.Series([False] * len(frame), index=frame.index)
    numeric_like = [
        feature for feature in features if feature not in CATEGORICAL_FEATURES
    ]
    present = frame[features].notna().sum(axis=1)
    mask = present >= 3
    if "DEMO_RIDAGEYR" in frame.columns:
        age = pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce")
        mask &= age.notna() & (age >= 18) & (age <= 120)
    if numeric_like:
        mask &= frame[numeric_like].notna().sum(axis=1) >= 2
    return mask


def build_intake_report(frame: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Screening + schema + missingness + eligibility, with no model inference."""
    screening = screen_dataset(frame)
    coverage = _coverage(frame, screening.mapped_features)
    violations = _range_violations(frame, screening.mapped_features)
    eligibility = _eligibility(frame, screening, coverage, violations)
    mask = eligible_row_mask(frame, screening.mapped_features)
    accepted = int(mask.sum())
    capabilities = dataset_capabilities(frame)
    usable = [
        feature
        for feature in screening.mapped_features
        if eligibility[feature]["tier"] in {"usable_now", "qualified_use"}
    ]
    return {
        "file": {"name": filename, "rows": int(len(frame)), "columns": int(frame.shape[1])},
        "limits": {
            "max_bytes": config.MAX_UPLOAD_BYTES,
            "max_rows": config.MAX_UPLOAD_ROWS,
            "max_scored_rows": config.MAX_SCORED_ROWS,
        },
        "schema": {
            "mapped_features": screening.mapped_features,
            "missing_features": screening.missing_features,
            "unmapped_columns": screening.unmapped_columns[:100],
            "unmapped_column_count": len(screening.unmapped_columns),
            "prohibited_columns": screening.prohibited_columns,
            "allowlist": PREVENTION_FEATURES,
        },
        "preview": _preview(frame, screening.mapped_features),
        "missingness": {
            feature: round(1.0 - coverage.get(feature, 0.0), 6)
            for feature in screening.mapped_features
        },
        "coverage": coverage,
        "range_violations": violations,
        "feature_eligibility": eligibility,
        "tier_definitions": TIER_DEFINITIONS,
        "rows": {
            "total": int(len(frame)),
            "accepted": accepted,
            "rejected": int(len(frame) - accepted),
            "rejection_rule": (
                "A row is scoreable when adult age is present and plausible and at least "
                "three allowlisted features (two of them numeric) are populated."
            ),
            "will_be_scored": min(accepted, config.MAX_SCORED_ROWS),
            "row_cap_applied": accepted > config.MAX_SCORED_ROWS,
        },
        "dataset_capability": {
            "rows": capabilities.get("rows"),
            "has_repeated_patient_measurements": capabilities.get(
                "has_repeated_patient_measurements"
            ),
            "supports_future_development_prediction": capabilities.get(
                "supports_future_development_prediction"
            ),
            "supported_output": capabilities.get("supported_output"),
            "warning": capabilities.get("warning"),
            "clustering_available_in_deployment": False,
            "clustering_note": (
                "The validated clustering pipeline (bootstrap stability, seed stability and "
                "survey-cycle negative controls) is not run in this deployment. No cluster "
                "assignment is produced for uploaded data, and no cluster may be described "
                "as a disease, cancer type or subtype."
            ),
        },
        "model_ready": bool(usable) and accepted > 0,
        "blockers": (
            []
            if (usable and accepted > 0)
            else [
                "No allowlisted feature reached a usable tier."
                if not usable
                else "No row satisfied the minimum feature requirement."
            ]
        ),
    }


def analysis_budget_exceeded(started_at: float) -> bool:
    return (time.monotonic() - started_at) > config.ANALYSIS_TIME_BUDGET_SECONDS
