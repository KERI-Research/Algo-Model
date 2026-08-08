"""Endpoint protocol, labels, eligibility masks and split protocol for future-risk work.

Everything here is decided **before** any model sees the data, and persisted so a reviewer
can check that the protocol was not tuned afterwards:

* index date per patient (the prediction time),
* prevalent-disease exclusion (outcome already present at or before index),
* washout / minimum-history requirement,
* censoring, administrative end of follow-up and death as a competing event,
* 1/3/5-year labels with **explicit eligibility masks** - a patient censored before a
  horizon is never treated as an ordinary negative,
* IPCW-ready censoring metadata (time, indicator, cause),
* patient-level splits plus a calendar/temporal holdout by index year,
* split manifest with fingerprints written before preprocessing.

Type 1 diabetes stays disabled. Site-specific cancer heads stay disabled unless the site
passes the 50-event gate, which is checked here and reported, not assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from longitudinal_schema import (
    CapabilityState,
    HORIZON_DAYS,
    HORIZON_LABELS,
    MIN_EVENTS_PER_HORIZON,
    MIN_NON_EVENTS_PER_HORIZON,
    PREVENTION_SAFE_FEATURES,
    SUPPORTED_OUTCOMES,
    assert_outcome_allowed,
    build_visit_matrix,
    frame_fingerprint,
    horizon_gate,
    validate_event_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EndpointProtocol:
    """The frozen endpoint definition. Changing it means a new dataset version."""

    protocol_version: str = "future-risk-protocol-v1"
    horizons_days: tuple[int, ...] = HORIZON_DAYS
    #: Minimum observed history before the index date (washout for prevalent disease).
    minimum_history_days: int = 365
    #: Minimum number of pre-index visits required to build a trajectory.
    minimum_visits: int = 2
    #: Outcomes modelled. Site heads are added only when their gate passes.
    outcomes: tuple[str, ...] = SUPPORTED_OUTCOMES
    #: Death is a competing event: it removes a patient from being an ordinary negative.
    competing_events: tuple[str, ...] = ("death",)
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    seed: int = 20260805
    #: Patients whose index year is at or after this year form the temporal holdout.
    temporal_holdout_from_year: int | None = 2016
    #: If the fixed calendar cutoff would hold out more than this share of the cohort, a
    #: cohort-relative cutoff is used instead and the substitution is recorded in the manifest.
    max_temporal_holdout_share: float = 0.25

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "horizons_days": list(self.horizons_days),
            "minimum_history_days": self.minimum_history_days,
            "minimum_visits": self.minimum_visits,
            "outcomes": list(self.outcomes),
            "competing_events": list(self.competing_events),
            "splits": {
                "train_fraction": self.train_fraction,
                "validation_fraction": self.validation_fraction,
                "test_fraction": round(1 - self.train_fraction - self.validation_fraction, 4),
                "seed": self.seed,
                "level": "patient",
                "temporal_holdout_from_year": self.temporal_holdout_from_year,
            },
            "label_rules": {
                "positive": "target outcome occurs within the horizon after the index date",
                "negative": "followed without the target outcome for at least the full horizon",
                "ineligible": (
                    "censored, died, or follow-up ended before the horizon without the target "
                    "outcome. Never scored as a negative."
                ),
            },
        }


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def _outcome_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: index date, first event per type, censoring, administrative end."""
    outcome_rows = events[events["outcome_type"] != "none"]
    index_dates = events.groupby("patient_id")["index_date"].max()
    last_observation = events.groupby("patient_id")["observation_timestamp"].max()

    table = pd.DataFrame({"index_date": index_dates, "last_record": last_observation})
    for outcome in ("type2_diabetes", "pan_cancer", "death"):
        selected = outcome_rows[outcome_rows["outcome_type"] == outcome]
        first = selected.groupby("patient_id")["event_date"].min()
        table[f"{outcome}_date"] = first
    sites = (
        outcome_rows[outcome_rows["outcome_type"] == "pan_cancer"]
        .sort_values("event_date")
        .groupby("patient_id")["cancer_site"]
        .first()
    )
    table["cancer_site"] = sites
    censoring = outcome_rows.groupby("patient_id")["censoring_date"].max()
    table["censoring_date"] = censoring
    table = table.reset_index()
    # Administrative end of follow-up: the censoring date when present, else the last record.
    table["follow_up_end"] = table["censoring_date"].fillna(table["last_record"])
    return table


def build_cohort(
    events: pd.DataFrame, protocol: EndpointProtocol | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the endpoint protocol and return the per-patient label frame plus a report."""
    protocol = protocol or EndpointProtocol()
    for outcome in protocol.outcomes:
        assert_outcome_allowed(outcome)

    observations = events[events["outcome_type"] == "none"]
    table = _outcome_table(events)

    history = observations.groupby("patient_id").agg(
        first_observation=("observation_timestamp", "min"),
        pre_index_visits=("visit_id", "nunique"),
    )
    table = table.merge(history, on="patient_id", how="left")
    table["history_days"] = (
        table["index_date"] - table["first_observation"]
    ).dt.total_seconds() / 86400.0

    exclusions: dict[str, int] = {}
    total = len(table)

    # Prevalent disease at or before the index date: excluded from the incident question.
    prevalent = pd.Series(False, index=table.index)
    for outcome in protocol.outcomes:
        column = f"{outcome}_date"
        prevalent_outcome = table[column].notna() & (table[column] <= table["index_date"])
        exclusions[f"prevalent_{outcome}"] = int(prevalent_outcome.sum())
        prevalent = prevalent | prevalent_outcome
    table["excluded_prevalent"] = prevalent

    table["excluded_short_history"] = table["history_days"].fillna(-1) < protocol.minimum_history_days
    table["excluded_few_visits"] = table["pre_index_visits"].fillna(0) < protocol.minimum_visits
    exclusions["short_history"] = int(table["excluded_short_history"].sum())
    exclusions["insufficient_visits"] = int(table["excluded_few_visits"].sum())

    eligible = ~(table["excluded_prevalent"] | table["excluded_short_history"] | table["excluded_few_visits"])
    cohort = table[eligible].copy().reset_index(drop=True)

    cohort["follow_up_days"] = (
        cohort["follow_up_end"] - cohort["index_date"]
    ).dt.total_seconds() / 86400.0
    cohort["index_year"] = cohort["index_date"].dt.year

    # Cause-specific time-to-event with death as a competing cause.
    for outcome in protocol.outcomes:
        event_days = (cohort[f"{outcome}_date"] - cohort["index_date"]).dt.total_seconds() / 86400.0
        death_days = (cohort["death_date"] - cohort["index_date"]).dt.total_seconds() / 86400.0
        cohort[f"{outcome}_event_days"] = event_days
        observed = event_days.notna() & (event_days <= cohort["follow_up_days"])
        death_first = (
            death_days.notna()
            & (~observed | (death_days < event_days.fillna(np.inf)))
            & (death_days <= cohort["follow_up_days"])
        )
        # IPCW-ready: time = min(event, death, censor); cause = 1 target, 2 competing, 0 censored.
        time_to = np.where(
            observed,
            event_days,
            np.where(death_first, death_days, cohort["follow_up_days"]),
        )
        cause = np.where(observed, 1, np.where(death_first, 2, 0))
        cohort[f"{outcome}_time_days"] = time_to
        cohort[f"{outcome}_cause"] = cause

        for horizon in protocol.horizons_days:
            suffix = HORIZON_LABELS[horizon]
            label = np.where(observed & (event_days <= horizon), 1, 0)
            # Eligible if the target event happened inside the horizon, or the patient was
            # followed event-free (and alive) for the entire horizon.
            followed_full = (time_to >= horizon) & (cause != 1)
            eligible_mask = ((label == 1) | followed_full).astype(int)
            cohort[f"{outcome}_{suffix}_label"] = label
            cohort[f"{outcome}_{suffix}_eligible"] = eligible_mask
            cohort[f"{outcome}_{suffix}_censored_before_horizon"] = (
                (label == 0) & (cause == 0) & (time_to < horizon)
            ).astype(int)
            cohort[f"{outcome}_{suffix}_competing_death_before_horizon"] = (
                (label == 0) & (cause == 2) & (time_to < horizon)
            ).astype(int)

    site_counts = cohort["cancer_site"].dropna().value_counts().to_dict()
    site_gates = {
        str(site): {
            "events": int(count),
            "enabled": bool(count >= MIN_EVENTS_PER_HORIZON),
            "reason": None if count >= MIN_EVENTS_PER_HORIZON else "below the 50-event gate",
        }
        for site, count in site_counts.items()
    }

    report = {
        "protocol": protocol.as_dict(),
        "patients_before_exclusions": int(total),
        "patients_in_cohort": int(len(cohort)),
        "exclusions": exclusions,
        "follow_up_days": {
            "median": round(float(cohort["follow_up_days"].median()), 2),
            "p10": round(float(cohort["follow_up_days"].quantile(0.10)), 2),
            "p90": round(float(cohort["follow_up_days"].quantile(0.90)), 2),
            "max": round(float(cohort["follow_up_days"].max()), 2),
        },
        "index_year_counts": {str(k): int(v) for k, v in cohort["index_year"].value_counts().sort_index().items()},
        "gates": {outcome: horizon_gate(cohort, outcome, protocol.horizons_days) for outcome in protocol.outcomes},
        "cancer_site_gates": site_gates,
        "site_heads_enabled": [site for site, item in site_gates.items() if item["enabled"]],
        "disabled_heads": {
            "type1_diabetes": "permanently disabled: needs autoantibodies/C-peptide/approved genetics",
            "pancreatic_cancer": "site head disabled until a real cohort passes the site gate",
        },
    }
    return cohort, report


# ---------------------------------------------------------------------------
# Splits (persisted before preprocessing)
# ---------------------------------------------------------------------------


def build_splits(
    cohort: pd.DataFrame, protocol: EndpointProtocol | None = None
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Patient-level random splits plus a calendar holdout defined by index year."""
    protocol = protocol or EndpointProtocol()
    patients = np.array(sorted(cohort["patient_id"].unique()))
    random = np.random.default_rng(protocol.seed)
    shuffled = random.permutation(patients)
    train_end = int(len(shuffled) * protocol.train_fraction)
    validation_end = train_end + int(len(shuffled) * protocol.validation_fraction)
    splits = {
        "train": sorted(shuffled[:train_end].tolist()),
        "validation": sorted(shuffled[train_end:validation_end].tolist()),
        "test": sorted(shuffled[validation_end:].tolist()),
    }
    holdout_rule = None
    if protocol.temporal_holdout_from_year is not None:
        cutoff_year = protocol.temporal_holdout_from_year
        candidate = cohort["index_year"] >= cutoff_year
        share = float(candidate.mean()) if len(cohort) else 0.0
        holdout_rule = f"fixed index year >= {cutoff_year}"
        # A fixed calendar cutoff is meaningless when almost every index date is after it
        # (Synthea index dates cluster in recent years). In that case fall back to a
        # cohort-relative cutoff so a usable training set survives, and record which rule was
        # used instead of silently producing a 9%-train split.
        if share > protocol.max_temporal_holdout_share:
            quantile = 1.0 - protocol.max_temporal_holdout_share
            # Year granularity is too coarse when index dates cluster inside one or two years,
            # so the cohort-relative cutoff is taken on the index *date* quantile.
            index_dates = pd.to_datetime(cohort["index_date"], errors="coerce", utc=True)
            cutoff = index_dates.quantile(quantile)
            candidate = index_dates >= cutoff
            holdout_rule = (
                f"cohort-relative cutoff: index date >= {cutoff.isoformat()} "
                f"(the {quantile:.0%} index-date quantile; the fixed "
                f"{protocol.temporal_holdout_from_year} year cutoff would have held out "
                f"{share:.1%} of the cohort, above the "
                f"{protocol.max_temporal_holdout_share:.0%} cap)"
            )
        temporal = sorted(cohort.loc[candidate, "patient_id"].tolist())
        splits["temporal_holdout"] = temporal
        # Random splits must not train on temporally held-out patients.
        temporal_set = set(temporal)
        splits["train"] = [p for p in splits["train"] if p not in temporal_set]
        splits["validation"] = [p for p in splits["validation"] if p not in temporal_set]
        splits["test"] = [p for p in splits["test"] if p not in temporal_set]

    overlaps = {}
    for first in ("train", "validation", "test"):
        for second in ("train", "validation", "test", "temporal_holdout"):
            if first >= second:
                continue
            if second in splits:
                overlaps[f"{first}__{second}"] = len(set(splits[first]) & set(splits[second]))
    if any(overlaps.values()):
        raise AssertionError(f"Patient overlap between splits: {overlaps}")

    manifest = {
        "split_manifest_version": "future-risk-splits-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "level": "patient",
        "seed": protocol.seed,
        "fractions": {
            "train": protocol.train_fraction,
            "validation": protocol.validation_fraction,
            "test": round(1 - protocol.train_fraction - protocol.validation_fraction, 4),
        },
        "temporal_holdout_from_year": protocol.temporal_holdout_from_year,
        "temporal_holdout_rule": holdout_rule,
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "overlaps": overlaps,
        "patient_id_fingerprints": {
            name: frame_fingerprint(pd.DataFrame({"patient_id": ids})) for name, ids in splits.items()
        },
        "note": (
            "Written before any preprocessing is fit. Preprocessing statistics are fit on "
            "train only; the temporal holdout is disjoint from train and validation."
        ),
    }
    return splits, manifest


# ---------------------------------------------------------------------------
# Feature assembly from the visit matrix
# ---------------------------------------------------------------------------

TRAJECTORY_SUFFIXES = ("last", "mean", "slope_per_year", "delta", "observed_count")


def build_patient_features(
    visit_matrix: pd.DataFrame, cohort: pd.DataFrame, features: tuple[str, ...] = PREVENTION_SAFE_FEATURES
) -> pd.DataFrame:
    """Summarise each patient's pre-index trajectory into tabular baseline features.

    These are the inputs for the transparent baselines. The temporal model consumes the
    visit sequence directly. Nothing after the index date is used.
    """
    rows: list[dict[str, Any]] = []
    grouped = visit_matrix.groupby("patient_id", sort=True)
    for patient_id, group in grouped:
        group = group.sort_values("visit_index")
        row: dict[str, Any] = {"patient_id": patient_id}
        row["visit_count"] = int(len(group))
        row["visit_density_per_year"] = float(group["visit_density_per_year"].iloc[-1])
        row["history_days"] = float(-group["relative_time_days"].min())
        row["median_visit_gap_days"] = float(group["delta_days_since_previous_visit"].iloc[1:].median() or 0.0)
        row["missingness_burden"] = float(
            1.0 - group[[f"mask_{feature}" for feature in features]].to_numpy().mean()
        )
        for feature in features:
            values = group[f"feature_{feature}"].astype(float)
            times = group["relative_time_days"].astype(float) / 365.25
            observed = values.notna()
            row[f"{feature}_observed_count"] = int(observed.sum())
            if observed.sum() == 0:
                row[f"{feature}_last"] = np.nan
                row[f"{feature}_mean"] = np.nan
                row[f"{feature}_slope_per_year"] = np.nan
                row[f"{feature}_delta"] = np.nan
                continue
            row[f"{feature}_last"] = float(values[observed].iloc[-1])
            row[f"{feature}_mean"] = float(values[observed].mean())
            row[f"{feature}_delta"] = float(values[observed].iloc[-1] - values[observed].iloc[0])
            if observed.sum() >= 2 and times[observed].std() > 0:
                slope = np.polyfit(times[observed], values[observed], 1)[0]
                row[f"{feature}_slope_per_year"] = float(slope)
            else:
                row[f"{feature}_slope_per_year"] = 0.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    return cohort.merge(frame, on="patient_id", how="inner")


def dataset_validation_report(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    visit_matrix: pd.DataFrame,
    protocol: EndpointProtocol,
) -> dict[str, Any]:
    """Coverage, missingness, event counts, follow-up, class balance and timeline checks."""
    feature_columns = [c for c in features.columns if c.endswith("_last")]
    timeline_problems: list[str] = []
    if (visit_matrix["relative_time_days"] > 0).any():
        timeline_problems.append("visit matrix contains post-index visits")
    merged = visit_matrix.merge(cohort[["patient_id", "index_date"]], on="patient_id", how="inner")
    if (merged["visit_timestamp"] > merged["index_date_y"] if "index_date_y" in merged else False).any():
        timeline_problems.append("visit timestamp after index date")
    for outcome in protocol.outcomes:
        negative_days = cohort[f"{outcome}_time_days"] < 0
        if negative_days.any():
            timeline_problems.append(f"{outcome}: negative time-to-event for {int(negative_days.sum())} patients")

    balance: dict[str, Any] = {}
    for outcome in protocol.outcomes:
        per_horizon = {}
        for horizon in protocol.horizons_days:
            suffix = HORIZON_LABELS[horizon]
            eligible = cohort[f"{outcome}_{suffix}_eligible"] == 1
            per_horizon[suffix] = {
                "eligible_patients": int(eligible.sum()),
                "events": int(cohort.loc[eligible, f"{outcome}_{suffix}_label"].sum()),
                "non_events": int((cohort.loc[eligible, f"{outcome}_{suffix}_label"] == 0).sum()),
                "prevalence_among_eligible": round(
                    float(cohort.loc[eligible, f"{outcome}_{suffix}_label"].mean()), 6
                )
                if eligible.any()
                else None,
                "censored_before_horizon": int(cohort[f"{outcome}_{suffix}_censored_before_horizon"].sum()),
                "competing_death_before_horizon": int(
                    cohort[f"{outcome}_{suffix}_competing_death_before_horizon"].sum()
                ),
            }
        balance[outcome] = per_horizon

    return {
        "report_type": "longitudinal_dataset_validation",
        "generated_at": datetime.now(UTC).isoformat(),
        "capability_state": CapabilityState.SIMULATION_ONLY_LONGITUDINAL.value,
        "simulation_only": True,
        "event_rows": int(len(events)),
        "patients_in_cohort": int(len(cohort)),
        "visit_rows": int(len(visit_matrix)),
        "visits_per_patient": {
            "median": float(visit_matrix.groupby("patient_id").size().median()),
            "min": int(visit_matrix.groupby("patient_id").size().min()),
            "max": int(visit_matrix.groupby("patient_id").size().max()),
        },
        "feature_coverage_last_value": {
            column: round(float(features[column].notna().mean()), 6) for column in feature_columns
        },
        "missingness_burden": {
            "mean": round(float(features["missingness_burden"].mean()), 6),
            "p90": round(float(features["missingness_burden"].quantile(0.9)), 6),
        },
        "follow_up_days": {
            "median": round(float(cohort["follow_up_days"].median()), 2),
            "iqr": [
                round(float(cohort["follow_up_days"].quantile(0.25)), 2),
                round(float(cohort["follow_up_days"].quantile(0.75)), 2),
            ],
        },
        "class_balance": balance,
        "timeline_checks": {
            "problems": timeline_problems,
            "status": "ok" if not timeline_problems else "blocked",
        },
        "gates": {outcome: horizon_gate(cohort, outcome, protocol.horizons_days) for outcome in protocol.outcomes},
        "minimum_gate": {"events": MIN_EVENTS_PER_HORIZON, "non_events": MIN_NON_EVENTS_PER_HORIZON},
    }


def prepare_dataset(
    events_path: str | Path,
    output_dir: str | Path,
    protocol: EndpointProtocol | None = None,
) -> dict[str, Any]:
    """Full protocol application: validate events, build cohort, features, splits, reports."""
    protocol = protocol or EndpointProtocol()
    events = pd.read_csv(events_path, low_memory=False)
    events, _ = validate_event_frame(events, dataset_name=Path(events_path).name, strict=True)
    cohort, cohort_report = build_cohort(events, protocol)
    visit_matrix = build_visit_matrix(events)
    visit_matrix = visit_matrix[visit_matrix["patient_id"].isin(cohort["patient_id"])].reset_index(drop=True)
    features = build_patient_features(visit_matrix, cohort)
    splits, split_manifest = build_splits(cohort, protocol)
    validation = dataset_validation_report(events, cohort, features, visit_matrix, protocol)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "cohort_labels.csv", index=False)
    features.to_csv(output / "patient_features.csv", index=False)
    visit_matrix.to_csv(output / "visit_matrix.csv", index=False)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    (output / "cohort_report.json").write_text(json.dumps(cohort_report, indent=2, default=str))
    (output / "dataset_validation_report.json").write_text(json.dumps(validation, indent=2, default=str))

    if validation["timeline_checks"]["status"] != "ok":
        raise ValueError(f"Timeline checks failed: {validation['timeline_checks']['problems']}")

    return {
        "output_dir": str(output),
        "patients_in_cohort": int(len(cohort)),
        "visit_rows": int(len(visit_matrix)),
        "feature_columns": int(features.shape[1]),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "gates": {outcome: validation["gates"][outcome]["per_horizon"] for outcome in protocol.outcomes},
        "site_heads_enabled": cohort_report["site_heads_enabled"],
        "capability_state": CapabilityState.SIMULATION_ONLY_LONGITUDINAL.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        default=str(PROJECT_ROOT / "data" / "synthetic_longitudinal" / "patient_events.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "synthetic_longitudinal" / "prepared"),
    )
    parser.add_argument("--minimum-history-days", type=int, default=365)
    parser.add_argument("--temporal-holdout-from-year", type=int, default=2016)
    arguments = parser.parse_args()
    protocol = EndpointProtocol(
        minimum_history_days=arguments.minimum_history_days,
        temporal_holdout_from_year=arguments.temporal_holdout_from_year,
    )
    print(json.dumps(prepare_dataset(arguments.events, arguments.output_dir, protocol), indent=2))


if __name__ == "__main__":
    main()