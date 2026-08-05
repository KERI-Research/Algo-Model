"""Data-integrity, leakage and capability validation for MetaboGuard.

This module is the single source of truth for:

* which dataset files are valid (and which are invalidated),
* how the NHANES cancer-site coding must be interpreted
  (MCQ230A-D: code 29 = Pancreas, code 39 = Other),
* which columns may be used as model inputs (allowlist) and which may never be
  (denylist: outcome labels and post-diagnosis TCGA context),
* deterministic, group-aware split boundaries,
* what the current data can and cannot support (capability gates).

Run as a CLI to produce a machine-readable report::

    python data_integrity.py --dataset ../data/nhanes_multicycle_v2.csv \
        --output ../model_artifacts/reports/data_integrity.json

The validator is fail-closed: ``validate_dataset(..., strict=True)`` raises on
any blocking finding, so training and benchmarking scripts cannot silently run
on invalidated data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry: valid vs invalidated inputs
# ---------------------------------------------------------------------------

#: NHANES MCQ230A-D cancer-site codes that matter for this project.
#: Source: NHANES Medical Conditions questionnaire codebooks (MCQ230A-D).
NHANES_CANCER_SITE_CODES: dict[int, str] = {29: "Pancreas", 39: "Other"}
PANCREAS_SITE_CODE = 29
OTHER_SITE_CODE = 39
MCQ230_COLUMNS = ("MCQ_MCQ230A", "MCQ_MCQ230B", "MCQ_MCQ230C", "MCQ_MCQ230D")

#: Datasets that must never be used again, and why.
INVALIDATED_DATASETS: dict[str, str] = {
    "nhanes_merged.csv": (
        "Pre-Priority-A schema and incorrect pancreatic-cancer coding "
        "(MCQ230 code 39 'Other' treated as pancreas). Use nhanes_merged_v2.csv."
    ),
    "nhanes_multicycle.csv": (
        "Incorrect pancreatic-cancer coding (MCQ230 code 39 'Other' treated as "
        "pancreas). Use nhanes_multicycle_v2.csv."
    ),
}

#: Supervised targets that are permanently disabled on current data.
INVALIDATED_TARGETS: dict[str, str] = {
    "PancreaticCancer": (
        "Historical supervised pancreatic-cancer results were invalidated by the "
        "MCQ230 29/39 correction. Corrected data hold only 19 prevalent cases, "
        "far below the 50-event safety gate, and they are prevalent (not "
        "incident) cases. Re-enabling is blocked until an incident cohort exists."
    ),
    "NODM_PancreaticCancer": (
        "New-onset-diabetes pancreatic-cancer target has 2 corrected positives. "
        "Permanently gated off on cross-sectional NHANES."
    ),
}

#: Model-input allowlist for prevention/deviation work (kept in sync with
#: self_supervised.PREVENTION_FEATURES; imported lazily to avoid a cycle).
def prevention_allowlist() -> list[str]:
    from self_supervised import PREVENTION_FEATURES

    return list(PREVENTION_FEATURES)


#: Columns that must never be model inputs for prevention scoring.
#: Outcome labels, label-derived variables, and every post-diagnosis TCGA field.
DENYLISTED_INPUT_COLUMNS: set[str] = {
    "Cancer",
    "MCQ_MCQ220",
    "MCQ_MCQ230A",
    "MCQ_MCQ230B",
    "MCQ_MCQ230C",
    "MCQ_MCQ230D",
    "MCQ_MCQ240T",
    "PancreaticCancer",
    "NODM_PancreaticCancer",
    "pancreatic_cancer_diagnosis_age",
    "pancreatic_cancer_minus_diabetes_years",
    "same_year_diabetes_pancreatic_cancer",
    "Diabetes",
    "DIQ_DIQ010",
    "diabetes_subtype",
    "new_onset_diabetes",
}

#: Any column with one of these prefixes is post-diagnosis context (TCGA) and is
#: denylisted for prevention scoring by construction.
DENYLISTED_INPUT_PREFIXES: tuple[str, ...] = ("tcga_",)

#: Label definitions used anywhere in the project, stated explicitly so that a
#: reviewer can check them against the code.
LABEL_DEFINITIONS: dict[str, str] = {
    "Cancer": "MCQ220 == 1 (self-reported ever told had cancer). PREVALENT, cross-sectional.",
    "PancreaticCancer": (
        "Any of MCQ230A-D == 29 (Pancreas). PREVALENT, cross-sectional. "
        "Code 39 is 'Other' and must never be counted here."
    ),
    "Diabetes": "DIQ010 == 1 (self-reported diagnosed diabetes). PREVALENT.",
    "diabetes_subtype": (
        "1 = research-only Type 1 proxy (young onset + insulin), 2 = Type 2 proxy, "
        "0 = no diabetes. Not a validated subtype: no autoantibodies, no approved "
        "genetics, no C-peptide-based confirmation in the current files."
    ),
}

#: Default intended future horizons and the minimum safety gate per horizon.
DEFAULT_HORIZON_DAYS: tuple[int, ...] = (365, 1095, 1825)
MIN_EVENTS_PER_HORIZON = 50
MIN_NON_EVENTS_PER_HORIZON = 50


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    code: str
    level: str  # "blocking" | "warning" | "info"
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "level": self.level, "message": self.message}


@dataclass
class IntegrityReport:
    dataset: str
    generated_at: str
    fingerprint: dict[str, Any] = field(default_factory=dict)
    row_counts: dict[str, Any] = field(default_factory=dict)
    identifiers: dict[str, Any] = field(default_factory=dict)
    cancer_coding: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    missingness: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    horizon_gates: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [item for item in self.findings if item.level == "blocking"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "status": "blocked" if self.blocking else "ok",
            "fingerprint": self.fingerprint,
            "row_counts": self.row_counts,
            "identifiers": self.identifiers,
            "cancer_coding": self.cancer_coding,
            "label_definitions": LABEL_DEFINITIONS,
            "labels": self.labels,
            "features": self.features,
            "missingness": self.missingness,
            "capabilities": self.capabilities,
            "horizon_gates": self.horizon_gates,
            "splits": self.splits,
            "findings": [item.as_dict() for item in self.findings],
        }


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """SHA-256 plus size/mtime, used to tie artifacts to exact input bytes."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "name": file_path.name,
        "sha256": digest.hexdigest(),
        "bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def assert_dataset_allowed(path: str | Path) -> None:
    """Fail closed when an invalidated dataset file is requested."""
    name = Path(path).name
    if name in INVALIDATED_DATASETS:
        raise ValueError(f"Dataset '{name}' is invalidated: {INVALIDATED_DATASETS[name]}")


def assert_target_allowed(target: str) -> None:
    """Fail closed on supervised targets that were invalidated by the correction."""
    if target in INVALIDATED_TARGETS:
        raise ValueError(
            f"Supervised target '{target}' is disabled: {INVALIDATED_TARGETS[target]}"
        )


def identifier_column(frame: pd.DataFrame) -> str | None:
    """Return the strongest available participant identifier."""
    for candidate in (
        "global_participant_id",
        "patient_id",
        "person_id",
        "subject_id",
        "bcr_patient_barcode",
        "SEQN",
    ):
        if candidate in frame.columns:
            return candidate
    return None


def is_denylisted_input(column: str) -> bool:
    return column in DENYLISTED_INPUT_COLUMNS or column.startswith(
        DENYLISTED_INPUT_PREFIXES
    )


def recompute_pancreatic_cancer(frame: pd.DataFrame) -> pd.Series | None:
    """Recompute the pancreatic-cancer label from MCQ230A-D using code 29."""
    present = [column for column in MCQ230_COLUMNS if column in frame.columns]
    if not present:
        return None
    codes = frame[present].apply(pd.to_numeric, errors="coerce")
    is_pancreas = (codes == PANCREAS_SITE_CODE).any(axis=1)
    if "Cancer" in frame.columns:
        cancer = pd.to_numeric(frame["Cancer"], errors="coerce")
        return pd.Series(
            np.where(is_pancreas, 1.0, np.where(cancer.notna(), 0.0, np.nan)),
            index=frame.index,
        )
    return is_pancreas.astype(float)


def validate_cancer_coding(frame: pd.DataFrame) -> tuple[dict[str, Any], list[Finding]]:
    """Verify that pancreas == code 29 and that code 39 ('Other') is not used."""
    findings: list[Finding] = []
    present = [column for column in MCQ230_COLUMNS if column in frame.columns]
    if not present:
        return (
            {"status": "not_applicable", "reason": "No MCQ230A-D columns in this file."},
            findings,
        )

    codes = frame[present].apply(pd.to_numeric, errors="coerce")
    pancreas_rows = int((codes == PANCREAS_SITE_CODE).any(axis=1).sum())
    other_rows = int((codes == OTHER_SITE_CODE).any(axis=1).sum())
    detail: dict[str, Any] = {
        "status": "checked",
        "site_code_meanings": {str(k): v for k, v in NHANES_CANCER_SITE_CODES.items()},
        "columns_checked": present,
        "rows_with_code_29_pancreas": pancreas_rows,
        "rows_with_code_39_other": other_rows,
    }

    recomputed = recompute_pancreatic_cancer(frame)
    if "PancreaticCancer" in frame.columns and recomputed is not None:
        stored = pd.to_numeric(frame["PancreaticCancer"], errors="coerce")
        stored_positives = int((stored == 1).sum())
        recomputed_positives = int((recomputed == 1).sum())
        detail["stored_positives"] = stored_positives
        detail["recomputed_positives_code_29"] = recomputed_positives
        detail["matches_code_29"] = stored_positives == recomputed_positives
        if stored_positives != recomputed_positives:
            findings.append(
                Finding(
                    "cancer_coding_mismatch",
                    "blocking",
                    "PancreaticCancer column does not match MCQ230A-D code 29 "
                    f"({stored_positives} stored vs {recomputed_positives} recomputed).",
                )
            )
        if stored_positives == other_rows and other_rows > 0:
            findings.append(
                Finding(
                    "cancer_coding_uses_code_39",
                    "blocking",
                    "PancreaticCancer positives equal the count of code 39 ('Other'). "
                    "This is the invalidated coding.",
                )
            )
    return detail, findings


def horizon_gate_report(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZON_DAYS,
) -> dict[str, Any]:
    """Evaluate the 50-event / 50-non-event gate for each intended horizon.

    Cross-sectional NHANES has no event time, so every horizon is reported as
    ungateable and future-risk heads stay disabled.
    """
    time_column = next(
        (
            column
            for column in frame.columns
            if column.lower()
            in {"event_time_days", "time_to_event_days", "followup_days"}
        ),
        None,
    )
    event_column = next(
        (column for column in frame.columns if column.lower() in {"event", "event_observed"}),
        None,
    )
    per_horizon: dict[str, Any] = {}
    if time_column is None or event_column is None:
        for horizon in horizons:
            per_horizon[f"{horizon}d"] = {
                "eligible": False,
                "events": 0,
                "non_events": 0,
                "reason": "No patient-level event time / event indicator columns exist.",
            }
        return {
            "minimum_events": MIN_EVENTS_PER_HORIZON,
            "minimum_non_events": MIN_NON_EVENTS_PER_HORIZON,
            "time_column": None,
            "event_column": None,
            "per_horizon": per_horizon,
            "any_horizon_eligible": False,
        }

    time = pd.to_numeric(frame[time_column], errors="coerce")
    status = pd.to_numeric(frame[event_column], errors="coerce")
    eligible_any = False
    for horizon in horizons:
        events = int(((status == 1) & (time <= horizon)).sum())
        non_events = int((time >= horizon).sum())
        eligible = (
            events >= MIN_EVENTS_PER_HORIZON and non_events >= MIN_NON_EVENTS_PER_HORIZON
        )
        eligible_any = eligible_any or eligible
        per_horizon[f"{horizon}d"] = {
            "eligible": eligible,
            "events": events,
            "non_events": non_events,
            "reason": None if eligible else "Below the 50/50 safety gate.",
        }
    return {
        "minimum_events": MIN_EVENTS_PER_HORIZON,
        "minimum_non_events": MIN_NON_EVENTS_PER_HORIZON,
        "time_column": time_column,
        "event_column": event_column,
        "per_horizon": per_horizon,
        "any_horizon_eligible": eligible_any,
    }


def group_split_indices(
    frame: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Deterministic participant-level train/validation/holdout split.

    Rows sharing a participant identifier always land in the same partition, so
    the split is patient-level whenever an identifier exists. Positional indices
    (not labels) are returned so callers can index transformed matrices safely.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    identifier = identifier_column(frame)
    positions = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    if identifier is None:
        order = rng.permutation(positions)
        groups_of = None
    else:
        keys = frame[identifier].astype(str).to_numpy()
        unique_keys = np.unique(keys)
        shuffled = rng.permutation(unique_keys)
        rank = {key: index for index, key in enumerate(shuffled)}
        order = positions[np.argsort([rank[key] for key in keys], kind="stable")]
        groups_of = keys
    train_end = int(len(order) * fractions[0])
    validation_end = train_end + int(len(order) * fractions[1])
    result = {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:validation_end]),
        "holdout": np.sort(order[validation_end:]),
    }
    if groups_of is not None:
        # Guarantee disjoint participants across partitions.
        seen: set[str] = set()
        for name in ("train", "validation", "holdout"):
            partition = set(groups_of[result[name]].tolist())
            overlap = seen & partition
            if overlap:
                raise AssertionError(
                    f"Participant overlap detected in split '{name}' ({len(overlap)} ids)."
                )
            seen |= partition
    return result


# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------


def validate_dataset(
    dataset_path: str | Path,
    strict: bool = True,
    minimum_adult_rows: int = 500,
) -> IntegrityReport:
    """Validate one dataset file and return a structured integrity report."""
    from self_supervised import dataset_capabilities

    path = Path(dataset_path)
    assert_dataset_allowed(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, low_memory=False)
    report = IntegrityReport(
        dataset=path.name, generated_at=datetime.now(UTC).isoformat()
    )
    report.fingerprint = file_fingerprint(path)

    age = pd.to_numeric(frame.get("DEMO_RIDAGEYR"), errors="coerce")
    adult_rows = int((age >= 18).sum()) if age is not None else 0
    report.row_counts = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "adult_rows_age_ge_18": adult_rows,
        "fully_duplicated_rows": int(frame.duplicated().sum()),
    }
    if adult_rows < minimum_adult_rows:
        report.findings.append(
            Finding(
                "insufficient_adult_rows",
                "blocking",
                f"Only {adult_rows} adult rows; at least {minimum_adult_rows} are required.",
            )
        )
    if report.row_counts["fully_duplicated_rows"]:
        report.findings.append(
            Finding(
                "duplicate_rows",
                "warning",
                f"{report.row_counts['fully_duplicated_rows']} fully duplicated rows present.",
            )
        )

    identifier = identifier_column(frame)
    duplicate_ids = (
        int(frame[identifier].duplicated().sum()) if identifier else None
    )
    report.identifiers = {
        "identifier_column": identifier,
        "unique_identifiers": int(frame[identifier].nunique()) if identifier else None,
        "duplicate_identifier_rows": duplicate_ids,
        "repeated_measurements": bool(duplicate_ids) if identifier else False,
        "note": (
            "NHANES is a repeated cross-section: each participant appears once, so "
            "participant-level and row-level splits coincide here. The split helper "
            "still groups by identifier so it stays correct if longitudinal data arrive."
        ),
    }
    if identifier is None:
        report.findings.append(
            Finding(
                "no_identifier",
                "warning",
                "No participant identifier column found; group-aware splitting degrades to row-level.",
            )
        )

    report.cancer_coding, coding_findings = validate_cancer_coding(frame)
    report.findings.extend(coding_findings)

    label_summary: dict[str, Any] = {}
    for label in ("Cancer", "PancreaticCancer", "Diabetes", "diabetes_subtype"):
        if label not in frame.columns:
            continue
        values = pd.to_numeric(frame[label], errors="coerce")
        label_summary[label] = {
            "positives": int((values == 1).sum()),
            "negatives": int((values == 0).sum()),
            "missing": int(values.isna().sum()),
        }
    if "diabetes_subtype" in frame.columns:
        subtype = pd.to_numeric(frame["diabetes_subtype"], errors="coerce")
        label_summary["type1_proxy_research_only"] = {
            "positives": int((subtype == 1).sum()),
            "validated": False,
            "reason": "No autoantibodies, no approved genetics, no confirmatory C-peptide criteria.",
        }
    report.labels = label_summary

    allowlist = prevention_allowlist()
    available = [column for column in allowlist if column in frame.columns]
    denylisted_present = sorted(
        column for column in frame.columns if is_denylisted_input(column)
    )
    leaked = sorted(set(available) & set(denylisted_present))
    report.features = {
        "allowlist_size": len(allowlist),
        "allowlist_available": available,
        "allowlist_missing": [c for c in allowlist if c not in frame.columns],
        "denylisted_columns_present_in_file": denylisted_present,
        "denylisted_columns_used_as_inputs": leaked,
        "post_diagnosis_prefixes": list(DENYLISTED_INPUT_PREFIXES),
    }
    if leaked:
        report.findings.append(
            Finding(
                "denylist_leak",
                "blocking",
                f"Denylisted columns appear in the model-input allowlist: {leaked}",
            )
        )

    report.missingness = {
        column: round(float(frame[column].isna().mean()), 6) for column in available
    }
    fully_missing = [c for c, rate in report.missingness.items() if rate >= 1.0]
    if fully_missing:
        report.findings.append(
            Finding(
                "fully_missing_features",
                "warning",
                f"Allowlisted features with no observed values: {fully_missing}",
            )
        )

    report.capabilities = dataset_capabilities(frame)
    report.horizon_gates = horizon_gate_report(frame)
    if not report.horizon_gates["any_horizon_eligible"]:
        report.findings.append(
            Finding(
                "no_future_risk_capability",
                "info",
                "No horizon passes the 50-event/50-non-event gate. Future-risk heads "
                "stay disabled; outputs are deviation/representation research only.",
            )
        )

    splits = group_split_indices(frame)
    report.splits = {
        "policy": "participant-grouped, seeded (seed=42), 70/15/15 train/validation/holdout",
        "sizes": {name: int(len(index)) for name, index in splits.items()},
        "preprocessing_fit_partition": "train only",
    }

    if strict and report.blocking:
        raise ValueError(
            "Blocking data-integrity findings: "
            + "; ".join(item.message for item in report.blocking)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="../data/nhanes_multicycle_v2.csv")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--allow-findings",
        action="store_true",
        help="Report blocking findings instead of exiting non-zero.",
    )
    arguments = parser.parse_args()

    report = validate_dataset(arguments.dataset, strict=False)
    payload = report.as_dict()
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "status": payload["status"],
                "rows": payload["row_counts"],
                "cancer_coding": payload["cancer_coding"],
                "capabilities": payload["capabilities"],
                "horizon_gates": payload["horizon_gates"]["per_horizon"],
                "findings": payload["findings"],
                "report_path": arguments.output,
            },
            indent=2,
        )
    )
    if report.blocking and not arguments.allow_findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()