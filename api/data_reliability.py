"""Structured data-reliability validation for MetaboGuard.

`data_integrity.py` answers "is this file allowed and correctly coded?".
This module answers the wider question a reviewer will ask: **how reliable is each
column, and what may we legitimately do with it today?**

It emits one machine-readable report covering:

* provenance and fingerprints,
* schema conformance,
* unit / range plausibility,
* duplicate participants,
* per-feature coverage,
* missingness by split, survey cycle and subgroup,
* assay / cycle drift (level and availability),
* label confidence,
* leakage controls,
* survey-weight applicability,
* capability state (what the data can and cannot support),
* **feature eligibility tiers**: ``usable_now``, ``qualified_use``, ``unavailable``,
  ``prohibited``.

The report fails closed: `build_reliability_report(..., strict=True)` raises on any hard
violation, and the CLI exits non-zero, so downstream analysis cannot run on data that
failed reliability review.

Plausibility bounds below are **project-set review thresholds**, chosen to catch encoding
errors and sentinel values. They are not clinical reference intervals and are not taken
from an external standard; they are deliberately wide.

CLI::

    python data_reliability.py --dataset ../data/nhanes_multicycle_v2.csv \
        --output ../model_artifacts/reports/data_reliability_nhanes_multicycle_v2.json
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

import data_integrity as di
from self_supervised import CATEGORICAL_FEATURES, dataset_capabilities

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Eligibility tiers, in decreasing order of usability.
TIERS = ("usable_now", "qualified_use", "unavailable", "prohibited")

#: Minimum non-missing fraction (over adults) for a feature to be `usable_now`.
MIN_COVERAGE_USABLE = 0.50
#: Coverage below this makes a feature `unavailable` in practice.
MIN_COVERAGE_QUALIFIED = 0.05
#: Fraction of implausible values tolerated before a feature is downgraded.
MAX_IMPLAUSIBLE_FRACTION = 0.01
#: Cycle-level median shift, in pooled IQR units, that flags assay/cycle drift.
DRIFT_IQR_THRESHOLD = 1.0
#: Fraction of cycles in which a feature must be measured at all.
MIN_CYCLE_AVAILABILITY = 0.60

#: Wide, project-set plausibility windows used only to catch encoding/sentinel errors.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "DEMO_RIDAGEYR": (0, 120),
    "BMX_BMXBMI": (8, 100),
    "BMX_BMXWAIST": (30, 220),
    "GHB_LBXGH": (2.5, 20),
    "GLU_LBXGLU": (20, 800),
    "INS_LBXIN": (0, 400),
    "CPEP_LBXCPSI": (0, 20),
    "TRIGLY_LBXTR": (10, 3000),
    "TRIGLY_LBDLDL": (10, 500),
    "HDL_LBDHDD": (5, 200),
    "TCHOL_LBXTC": (50, 600),
    "HSCRP_LBXHSCRP": (0, 200),
    "CBC_LBXHGB": (3, 25),
    "CBC_LBXPLTSI": (10, 1500),
    "BIOPRO_LBXSATSI": (1, 1000),
    "BIOPRO_LBXSAPSI": (5, 1000),
    "BIOPRO_LBXSCR": (0.1, 20),
    "homa_ir": (0, 100),
    "average_drinks_per_day": (0, 40),
    "weight_loss_1yr_lb": (-200, 300),
    "weight_loss_10yr_lb": (-300, 400),
}

#: Survey design columns. Their presence does not mean weights are being applied.
SURVEY_DESIGN_COLUMNS = (
    "DEMO_WTMEC2YR",
    "DEMO_WTMEC4YR",
    "DEMO_WTMECPRP",
    "DEMO_WTINT2YR",
    "DEMO_SDMVPSU",
    "DEMO_SDMVSTRA",
    "combined_mec_weight_1999_2020",
)

#: Reference documentation already cited in this repository.
REFERENCE_LINKS = {
    "nhanes_weighting_tutorial": "https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx",
    "nhanes_quality_guidelines": "https://wwwn.cdc.gov/nchs/nhanes/QualityAnalysesGuidelines.aspx",
    "sklearn_leakage_pitfalls": "https://scikit-learn.org/stable/common_pitfalls.html",
}


@dataclass
class Violation:
    code: str
    level: str  # "hard" | "soft"
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "level": self.level, "message": self.message}


@dataclass
class ReliabilityReport:
    dataset: str
    generated_at: str
    sections: dict[str, Any] = field(default_factory=dict)
    feature_eligibility: dict[str, Any] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def hard_violations(self) -> list[Violation]:
        return [item for item in self.violations if item.level == "hard"]

    def tier(self, tier_name: str) -> list[str]:
        return sorted(
            name
            for name, payload in self.feature_eligibility.items()
            if payload["tier"] == tier_name
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_type": "metaboguard_data_reliability",
            "report_schema_version": 1,
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "status": "blocked" if self.hard_violations else "ok",
            "tier_definitions": {
                "usable_now": (
                    f"Present, plausible, coverage >= {MIN_COVERAGE_USABLE:.0%} of adults, "
                    "no blocking drift or availability gap."
                ),
                "qualified_use": (
                    "Usable only with a stated caveat: low coverage, cycle availability gap, "
                    "assay/cycle drift, implausible-value burden, or a declared evidence gap."
                ),
                "unavailable": "Absent from this file, or effectively unmeasured.",
                "prohibited": "Denylisted as a model input (outcome label, label-derived, or post-diagnosis TCGA context).",
            },
            "tiers": {name: self.tier(name) for name in TIERS},
            "feature_eligibility": self.feature_eligibility,
            "sections": self.sections,
            "violations": [item.as_dict() for item in self.violations],
            "reference_links": REFERENCE_LINKS,
            "interpretation_note": (
                "This report is a data observation, not a model result and not a clinical "
                "statement. It describes the measurement quality of the input file only."
            ),
        }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _provenance(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    cycles = (
        sorted(frame["survey_cycle"].dropna().unique().tolist())
        if "survey_cycle" in frame.columns
        else []
    )
    return {
        "fingerprint": di.file_fingerprint(path),
        "registry_status": (
            "invalidated" if path.name in di.INVALIDATED_DATASETS else "allowed"
        ),
        "invalidated_reason": di.INVALIDATED_DATASETS.get(path.name),
        "survey_cycles_present": cycles,
        "survey_cycle_count": len(cycles),
        "source_description": (
            "NHANES public-release cycle files, pooled and harmonised by this repository's "
            "build scripts. TCGA-CDR is stored separately and is post-diagnosis context only."
        ),
        "external_upload_policy": "No row-level data leaves this machine.",
    }


def _schema(frame: pd.DataFrame, allowlist: list[str]) -> tuple[dict[str, Any], list[Violation]]:
    violations: list[Violation] = []
    dtypes = frame.dtypes.astype(str).to_dict()
    missing_allowlist = [column for column in allowlist if column not in frame.columns]
    numeric_expected = [
        column
        for column in allowlist
        if column in frame.columns and column not in CATEGORICAL_FEATURES
    ]
    non_numeric = [
        column
        for column in numeric_expected
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        violations.append(
            Violation(
                "schema_non_numeric_feature",
                "hard",
                f"Expected-numeric allowlist columns are not numeric: {non_numeric}",
            )
        )
    return (
        {
            "columns": int(frame.shape[1]),
            "rows": int(frame.shape[0]),
            "dtype_counts": {
                dtype: sum(1 for value in dtypes.values() if value == dtype)
                for dtype in sorted(set(dtypes.values()))
            },
            "allowlist_present": [c for c in allowlist if c in frame.columns],
            "allowlist_missing": missing_allowlist,
            "non_numeric_allowlist_columns": non_numeric,
            "identifier_column": di.identifier_column(frame),
        },
        violations,
    )


def _plausibility(adult: pd.DataFrame) -> tuple[dict[str, Any], list[Violation]]:
    violations: list[Violation] = []
    detail: dict[str, Any] = {}
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in adult.columns:
            continue
        values = pd.to_numeric(adult[column], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            detail[column] = {"observed": 0, "note": "no observed values"}
            continue
        out_of_range = int(((observed < low) | (observed > high)).sum())
        fraction = out_of_range / len(observed)
        detail[column] = {
            "bounds": [low, high],
            "observed": int(len(observed)),
            "out_of_range": out_of_range,
            "out_of_range_fraction": round(float(fraction), 6),
            "min": round(float(observed.min()), 4),
            "max": round(float(observed.max()), 4),
            "flag": fraction > MAX_IMPLAUSIBLE_FRACTION,
        }
        if fraction > 0.10:
            violations.append(
                Violation(
                    "implausible_value_burden",
                    "hard",
                    f"{column}: {fraction:.1%} of observed values fall outside the "
                    f"project plausibility window {low}-{high}. Suspect unit or sentinel error.",
                )
            )
    detail["_note"] = (
        "Bounds are project-set review thresholds for catching encoding and sentinel "
        "errors. They are not clinical reference intervals."
    )
    return detail, violations


def _duplicates(frame: pd.DataFrame) -> tuple[dict[str, Any], list[Violation]]:
    violations: list[Violation] = []
    detail: dict[str, Any] = {"fully_duplicated_rows": int(frame.duplicated().sum())}
    for column in ("SEQN", "global_participant_id"):
        if column in frame.columns:
            duplicated = int(frame[column].duplicated().sum())
            detail[column] = {
                "unique": int(frame[column].nunique()),
                "duplicated_rows": duplicated,
            }
            if column == "global_participant_id" and duplicated:
                detail[column]["interpretation"] = (
                    "Repeated participant identifiers imply repeated measurements: "
                    "grouped splitting becomes mandatory rather than merely defensive."
                )
    if detail["fully_duplicated_rows"] > 0:
        violations.append(
            Violation(
                "duplicate_rows",
                "soft",
                f"{detail['fully_duplicated_rows']} fully duplicated rows.",
            )
        )
    return detail, violations


def _coverage_and_missingness(
    adult: pd.DataFrame, allowlist: list[str], splits: dict[str, np.ndarray]
) -> dict[str, Any]:
    available = [column for column in allowlist if column in adult.columns]
    coverage = {
        column: round(float(adult[column].notna().mean()), 6) for column in available
    }
    by_split = {
        name: {
            column: round(float(adult.iloc[index][column].notna().mean()), 6)
            for column in available
        }
        for name, index in splits.items()
    }
    by_cycle: dict[str, Any] = {}
    if "survey_cycle" in adult.columns:
        for cycle, group in adult.groupby("survey_cycle"):
            by_cycle[str(cycle)] = {
                column: round(float(group[column].notna().mean()), 6) for column in available
            }
    subgroups: dict[str, Any] = {}
    if "DEMO_RIAGENDR" in adult.columns:
        for code, label in ((1, "male"), (2, "female")):
            group = adult[pd.to_numeric(adult["DEMO_RIAGENDR"], errors="coerce") == code]
            if len(group):
                subgroups[f"sex_{label}"] = {
                    "rows": int(len(group)),
                    "mean_feature_coverage": round(
                        float(np.mean([group[c].notna().mean() for c in available])), 6
                    ),
                }
    age = pd.to_numeric(adult.get("DEMO_RIDAGEYR"), errors="coerce")
    if age is not None:
        for label, mask in (
            ("age_18_39", (age >= 18) & (age < 40)),
            ("age_40_59", (age >= 40) & (age < 60)),
            ("age_60_plus", age >= 60),
        ):
            group = adult[mask.fillna(False)]
            if len(group):
                subgroups[label] = {
                    "rows": int(len(group)),
                    "mean_feature_coverage": round(
                        float(np.mean([group[c].notna().mean() for c in available])), 6
                    ),
                }
    missingness_burden = (
        adult[available].isna().sum(axis=1) / max(len(available), 1)
    )
    return {
        "features_evaluated": available,
        "coverage_overall": coverage,
        "coverage_by_split": by_split,
        "coverage_by_cycle": by_cycle,
        "coverage_by_subgroup": subgroups,
        "row_missingness_burden": {
            "mean": round(float(missingness_burden.mean()), 6),
            "p50": round(float(missingness_burden.quantile(0.5)), 6),
            "p90": round(float(missingness_burden.quantile(0.9)), 6),
            "rows_with_no_labs": int((missingness_burden >= 0.9).sum()),
        },
    }


def _drift(adult: pd.DataFrame, allowlist: list[str]) -> tuple[dict[str, Any], list[Violation]]:
    violations: list[Violation] = []
    if "survey_cycle" not in adult.columns:
        return {"status": "not_applicable", "reason": "no survey_cycle column"}, violations
    numeric = [
        column
        for column in allowlist
        if column in adult.columns and column not in CATEGORICAL_FEATURES
    ]
    cycles = sorted(adult["survey_cycle"].dropna().unique().tolist())
    detail: dict[str, Any] = {"cycles": [str(c) for c in cycles], "features": {}}
    for column in numeric:
        values = pd.to_numeric(adult[column], errors="coerce")
        pooled_iqr = float(values.quantile(0.75) - values.quantile(0.25))
        pooled_median = float(values.median()) if values.notna().any() else float("nan")
        if not np.isfinite(pooled_iqr) or pooled_iqr <= 0:
            continue
        per_cycle_median: dict[str, float | None] = {}
        availability: dict[str, float] = {}
        for cycle in cycles:
            group = values[adult["survey_cycle"] == cycle]
            availability[str(cycle)] = round(float(group.notna().mean()), 6)
            per_cycle_median[str(cycle)] = (
                round(float(group.median()), 6) if group.notna().any() else None
            )
        observed_medians = [m for m in per_cycle_median.values() if m is not None]
        max_shift = (
            max(abs(m - pooled_median) for m in observed_medians) / pooled_iqr
            if observed_medians
            else 0.0
        )
        cycles_measured = sum(1 for value in availability.values() if value > 0.01)
        availability_fraction = cycles_measured / max(len(cycles), 1)
        entry = {
            "pooled_median": round(pooled_median, 6),
            "pooled_iqr": round(pooled_iqr, 6),
            "median_by_cycle": per_cycle_median,
            "coverage_by_cycle": availability,
            "max_median_shift_in_iqr_units": round(float(max_shift), 4),
            "level_drift_flag": bool(max_shift > DRIFT_IQR_THRESHOLD),
            "cycles_measured": cycles_measured,
            "cycle_availability_fraction": round(float(availability_fraction), 4),
            "availability_gap_flag": bool(availability_fraction < MIN_CYCLE_AVAILABILITY),
        }
        detail["features"][column] = entry
    detail["level_drift_features"] = sorted(
        name for name, item in detail["features"].items() if item["level_drift_flag"]
    )
    detail["availability_gap_features"] = sorted(
        name for name, item in detail["features"].items() if item["availability_gap_flag"]
    )
    detail["note"] = (
        "Cycle-to-cycle shifts mix real population change, assay changes and subsample "
        "design. A flag means 'do not interpret this feature's absolute level across "
        "cycles without adjustment', not 'the data are wrong'."
    )
    return detail, violations


def _label_confidence(frame: pd.DataFrame) -> dict[str, Any]:
    def counts(column: str) -> dict[str, int]:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        if values is None:
            return {}
        return {
            "positives": int((values == 1).sum()),
            "negatives": int((values == 0).sum()),
            "missing": int(values.isna().sum()),
        }

    pancreatic = counts("PancreaticCancer")
    return {
        "source": "self-reported questionnaire items (MCQ220, MCQ230A-D, DIQ010)",
        "confidence": "low_to_moderate",
        "definitions": di.LABEL_DEFINITIONS,
        "counts": {
            "Cancer": counts("Cancer"),
            "PancreaticCancer": pancreatic,
            "Diabetes": counts("Diabetes"),
        },
        "site_assignment_supported": False,
        "site_assignment_reason": (
            "Cancer site comes from a self-reported multi-select item with small per-site "
            "counts; the pancreatic label holds "
            f"{pancreatic.get('positives', 0)} prevalent cases. Site-specific outputs stay disabled."
        ),
        "type1_proxy": {
            "research_only": True,
            "reason": "No autoantibodies, no approved genetics, no confirmatory C-peptide criteria.",
        },
        "prevalent_not_incident": True,
        "caveats": [
            "Recall and survivorship bias: fatal cancers are under-represented in a household survey.",
            "No diagnosis dates usable as event times, so no incidence can be derived.",
            "Labels may only characterise clusters post hoc; they never enter fitting or selection.",
        ],
    }


def _leakage(frame: pd.DataFrame, allowlist: list[str]) -> tuple[dict[str, Any], list[Violation]]:
    violations: list[Violation] = []
    denylisted_present = sorted(c for c in frame.columns if di.is_denylisted_input(c))
    leaked = sorted(set(allowlist) & set(denylisted_present))
    tcga_columns = sorted(c for c in frame.columns if c.startswith("tcga_"))
    if leaked:
        violations.append(
            Violation("denylist_leak", "hard", f"Denylisted columns in the allowlist: {leaked}")
        )
    if set(tcga_columns) & set(allowlist):
        violations.append(
            Violation(
                "tcga_post_diagnosis_leak",
                "hard",
                "Post-diagnosis TCGA columns reached the model-input allowlist.",
            )
        )
    return (
        {
            "denylisted_columns_present": denylisted_present,
            "denylisted_columns_in_allowlist": leaked,
            "tcga_post_diagnosis_columns_present": tcga_columns,
            "tcga_columns_in_allowlist": sorted(set(tcga_columns) & set(allowlist)),
            "split_policy": "participant-grouped, seeded; preprocessing fit on train only",
            "reference": REFERENCE_LINKS["sklearn_leakage_pitfalls"],
        },
        violations,
    )


def _survey_weights(frame: pd.DataFrame) -> dict[str, Any]:
    present = [column for column in SURVEY_DESIGN_COLUMNS if column in frame.columns]
    return {
        "design_columns_present": present,
        "weights_applied_in_modelling": False,
        "applicable_to": [
            "population prevalence estimates",
            "descriptive subgroup statistics intended to represent the US population",
        ],
        "not_applicable_to": [
            "unsupervised representation learning",
            "deviation percentiles (defined against the training reference sample)",
            "cluster discovery and cluster stability",
        ],
        "consequence": (
            "All model outputs describe the pooled analytic sample, not the US population. "
            "Any population-level claim requires MEC weights with PSU/strata variance "
            "estimation."
        ),
        "reference": REFERENCE_LINKS["nhanes_weighting_tutorial"],
    }


def _eligibility(
    frame: pd.DataFrame,
    allowlist: list[str],
    coverage: dict[str, Any],
    plausibility: dict[str, Any],
    drift: dict[str, Any],
) -> dict[str, Any]:
    """Assign an eligibility tier and reasons to every candidate and denylisted column."""
    from evidence_catalogue import load_catalogue

    try:
        catalogue = load_catalogue()
        evidence_gap_columns = {
            entry.get("current_data_column")
            for entry in catalogue.entries
            if str(entry.get("validation_status")) == "evidence_gap"
        }
        weak_evidence_columns = {
            entry.get("current_data_column")
            for entry in catalogue.entries
            if "no_association" in str(entry.get("evidence_grade"))
        }
    except Exception:
        evidence_gap_columns, weak_evidence_columns = set(), set()

    eligibility: dict[str, Any] = {}
    overall_coverage = coverage["coverage_overall"]
    drift_features = drift.get("features", {}) if isinstance(drift, dict) else {}

    for column in allowlist:
        reasons: list[str] = []
        if di.is_denylisted_input(column):
            eligibility[column] = {
                "tier": "prohibited",
                "reasons": ["Denylisted as a model input."],
            }
            continue
        if column not in frame.columns:
            eligibility[column] = {"tier": "unavailable", "reasons": ["Column absent from file."]}
            continue
        column_coverage = overall_coverage.get(column, 0.0)
        if column_coverage < MIN_COVERAGE_QUALIFIED:
            eligibility[column] = {
                "tier": "unavailable",
                "reasons": [f"Coverage {column_coverage:.1%} is effectively unmeasured."],
            }
            continue
        if column_coverage < MIN_COVERAGE_USABLE:
            reasons.append(
                f"Coverage {column_coverage:.1%} is below the {MIN_COVERAGE_USABLE:.0%} threshold "
                "(NHANES subsample design)."
            )
        entry = drift_features.get(column, {})
        if entry.get("availability_gap_flag"):
            reasons.append(
                f"Measured in only {entry.get('cycles_measured')} of "
                f"{len(drift.get('cycles', []))} cycles."
            )
        if entry.get("level_drift_flag"):
            reasons.append(
                "Cycle-level median shift exceeds "
                f"{DRIFT_IQR_THRESHOLD} IQR: absolute levels are not comparable across cycles."
            )
        plaus = plausibility.get(column, {})
        if isinstance(plaus, dict) and plaus.get("flag"):
            reasons.append(
                f"{plaus.get('out_of_range_fraction', 0):.2%} of values fall outside the "
                "project plausibility window."
            )
        if column in evidence_gap_columns:
            reasons.append("Declared evidence gap in the biomarker evidence catalogue.")
        if column in weak_evidence_columns:
            reasons.append(
                "Catalogued evidence indicates no consistent association: treat as a "
                "known-weak feature."
            )
        eligibility[column] = {
            "tier": "qualified_use" if reasons else "usable_now",
            "reasons": reasons or ["Present, plausible and adequately covered."],
            "coverage": round(float(column_coverage), 6),
        }

    for column in sorted(c for c in frame.columns if di.is_denylisted_input(c)):
        eligibility.setdefault(
            column,
            {
                "tier": "prohibited",
                "reasons": [
                    "Outcome label, label-derived, or post-diagnosis TCGA context: "
                    "never a model input for prevention or clustering."
                ],
            },
        )
    return eligibility


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_reliability_report(
    dataset_path: str | Path, strict: bool = False
) -> ReliabilityReport:
    path = Path(dataset_path)
    di.assert_dataset_allowed(path)
    frame = pd.read_csv(path, low_memory=False)
    allowlist = di.prevention_allowlist()

    report = ReliabilityReport(
        dataset=path.name, generated_at=datetime.now(UTC).isoformat()
    )
    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18].reset_index(
        drop=True
    )
    splits = di.group_split_indices(adult, seed=42)

    report.sections["provenance"] = _provenance(path, frame)
    schema, schema_violations = _schema(frame, allowlist)
    report.sections["schema"] = schema
    report.violations.extend(schema_violations)

    plausibility, plausibility_violations = _plausibility(adult)
    report.sections["unit_range_plausibility"] = plausibility
    report.violations.extend(plausibility_violations)

    duplicates, duplicate_violations = _duplicates(frame)
    report.sections["duplicate_participants"] = duplicates
    report.violations.extend(duplicate_violations)

    coverage = _coverage_and_missingness(adult, allowlist, splits)
    report.sections["coverage_and_missingness"] = coverage

    drift, drift_violations = _drift(adult, allowlist)
    report.sections["assay_cycle_drift"] = drift
    report.violations.extend(drift_violations)

    report.sections["label_confidence"] = _label_confidence(frame)

    leakage, leakage_violations = _leakage(frame, allowlist)
    report.sections["leakage_controls"] = leakage
    report.violations.extend(leakage_violations)

    report.sections["survey_weights"] = _survey_weights(frame)

    capabilities = dataset_capabilities(frame)
    report.sections["capability_state"] = {
        "capabilities": capabilities,
        "horizon_gates": di.horizon_gate_report(frame),
        "future_risk_enabled": False,
        "cancer_site_assignment_enabled": False,
        "clustering_enabled": True,
        "clustering_scope": (
            "Label-free discovery of patient/metabolic phenotypes. Clusters are never "
            "cancer diagnoses, subtypes or sites."
        ),
    }
    report.sections["row_counts"] = {
        "rows": int(len(frame)),
        "adult_rows": int(len(adult)),
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
    }

    report.feature_eligibility = _eligibility(frame, allowlist, coverage, plausibility, drift)

    if not report.tier("usable_now"):
        report.violations.append(
            Violation(
                "no_usable_features",
                "hard",
                "No feature reached the usable_now tier; analysis cannot proceed.",
            )
        )

    if strict and report.hard_violations:
        raise ValueError(
            "Data reliability hard violations: "
            + "; ".join(item.message for item in report.hard_violations)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()

    report = build_reliability_report(arguments.dataset, strict=arguments.strict)
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
                "row_counts": payload["sections"]["row_counts"],
                "tiers": {name: len(items) for name, items in payload["tiers"].items()},
                "qualified_use": payload["tiers"]["qualified_use"],
                "unavailable": payload["tiers"]["unavailable"],
                "level_drift_features": payload["sections"]["assay_cycle_drift"].get(
                    "level_drift_features"
                ),
                "availability_gap_features": payload["sections"]["assay_cycle_drift"].get(
                    "availability_gap_features"
                ),
                "violations": payload["violations"],
                "report_path": arguments.output,
            },
            indent=2,
        )
    )
    if report.hard_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()