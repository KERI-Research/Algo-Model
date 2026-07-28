"""Leave-one-NHANES-cycle-out temporal validation for MetaboGuard.

Each survey cycle is held out in turn. All preprocessing statistics and model
parameters are learned using the remaining cycles only. Two feature variants
are compared:

1. clinical_only: excludes survey-cycle and cohort-standardised HbA1c proxies.
2. with_cycle_proxies: uses the full selected feature set.

The output is a machine-readable JSON report and a source-cited Markdown
summary. This validation tests temporal transportability across repeated
cross-sections; it is not external validation on a different data source.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from biomarker import (
    _build_candidate_models,
    _prepare_dataframe,
    _resolve_field_sets,
    _select_features,
)


TEMPORAL_PROXY_FEATURES = {
    "survey_cycle_index",
    "hba1c_cycle_age_sex_z",
}


def _benchmark_models(positive_rate: float) -> list[tuple[str, Any]]:
    """Classical baselines plus MetaboGuard's two boosted-tree candidates."""
    models: list[tuple[str, Any]] = [
        (
            "logistic_regression_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
    models.extend(_build_candidate_models(positive_rate=positive_rate))
    return models


@dataclass
class FoldResult:
    variant: str
    model: str
    held_out_cycle: str
    train_rows: int
    test_rows: int
    train_positives: int
    test_positives: int
    test_prevalence: float
    auroc: float
    auprc: float
    auprc_lift: float
    brier_score: float


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(y.mean())
    auroc = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")
    auprc = float(average_precision_score(y, probability))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": auprc / max(prevalence, 1e-12),
        "brier_score": float(brier_score_loss(y, probability)),
        "prevalence": prevalence,
    }


def _cycle_cluster_bootstrap(
    predictions: list[dict[str, Any]],
    repeats: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap whole held-out cycles, preserving within-cycle dependence."""
    if len(predictions) < 3:
        return {"auroc_ci_95": [float("nan"), float("nan")],
                "auprc_ci_95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    for _ in range(repeats):
        selected = rng.choice(len(predictions), size=len(predictions), replace=True)
        y = np.concatenate([predictions[i]["y"] for i in selected])
        probability = np.concatenate([predictions[i]["probability"] for i in selected])
        if len(np.unique(y)) < 2:
            continue
        auroc_values.append(float(roc_auc_score(y, probability)))
        auprc_values.append(float(average_precision_score(y, probability)))
    return {
        "auroc_ci_95": [round(float(x), 6) for x in np.quantile(auroc_values, [0.025, 0.975])],
        "auprc_ci_95": [round(float(x), 6) for x in np.quantile(auprc_values, [0.025, 0.975])],
    }


def _evaluate_configuration(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    variant: str,
) -> tuple[list[FoldResult], dict[str, Any]]:
    folds: list[FoldResult] = []
    predictions_by_model: dict[str, list[dict[str, Any]]] = {}
    cycles = sorted(frame["survey_cycle"].dropna().unique().tolist())

    for held_out_cycle in cycles:
        train = frame[frame["survey_cycle"] != held_out_cycle].copy()
        test = frame[frame["survey_cycle"] == held_out_cycle].copy()
        train = train.dropna(subset=[target])
        test = test.dropna(subset=[target])
        train = train[train[target].isin([0, 1])]
        test = test[test[target].isin([0, 1])]
        # Training requires both classes. Keep single-class test cycles in the
        # pooled out-of-cycle predictions; excluding zero-event cycles would
        # bias prevalence and AUPRC upward. Fold AUROC is NaN when one class is
        # absent, which is the honest result.
        if train[target].nunique() < 2 or test.empty:
            continue

        x_train = train[features].apply(pd.to_numeric, errors="coerce")
        x_test = test[features].apply(pd.to_numeric, errors="coerce")
        y_train = train[target].to_numpy(dtype=int)
        y_test = test[target].to_numpy(dtype=int)

        # Leakage-safe preprocessing: train-cycle medians only.
        medians = x_train.median(numeric_only=True)
        x_train = x_train.fillna(medians)
        x_test = x_test.fillna(medians)
        positive_rate = float(y_train.mean())

        for model_name, model in _benchmark_models(positive_rate=positive_rate):
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            metric = _metrics(y_test, probability)
            folds.append(
                FoldResult(
                    variant=variant,
                    model=model_name,
                    held_out_cycle=held_out_cycle,
                    train_rows=len(train),
                    test_rows=len(test),
                    train_positives=int((y_train == 1).sum()),
                    test_positives=int((y_test == 1).sum()),
                    test_prevalence=round(metric["prevalence"], 8),
                    auroc=round(metric["auroc"], 6),
                    auprc=round(metric["auprc"], 6),
                    auprc_lift=round(metric["auprc_lift"], 6),
                    brier_score=round(metric["brier_score"], 6),
                )
            )
            predictions_by_model.setdefault(model_name, []).append(
                {"cycle": held_out_cycle, "y": y_test, "probability": probability}
            )

    pooled: dict[str, Any] = {}
    for model_name, predictions in predictions_by_model.items():
        y = np.concatenate([item["y"] for item in predictions])
        probability = np.concatenate([item["probability"] for item in predictions])
        metric = _metrics(y, probability)
        pooled[model_name] = {
            "rows": int(len(y)),
            "positives": int((y == 1).sum()),
            **{key: round(value, 6) for key, value in metric.items()},
            **_cycle_cluster_bootstrap(predictions),
        }
    return folds, pooled


def validate(
    dataset: Path,
    target: str = "PancreaticCancer",
    cohort_filter: str | None = "diabetics_only",
) -> dict[str, Any]:
    frame = _prepare_dataframe(pd.read_csv(dataset, low_memory=False))
    if cohort_filter == "diabetics_only":
        frame = frame[pd.to_numeric(frame["Diabetes"], errors="coerce") == 1].copy()
    if "survey_cycle" not in frame.columns:
        raise ValueError("Cycle-held-out validation requires survey_cycle.")

    required, _ = _resolve_field_sets(frame)
    frame = frame.dropna(subset=required)
    features = [f for f in _select_features(frame) if f != target]
    variants = {
        "clinical_only": [f for f in features if f not in TEMPORAL_PROXY_FEATURES],
        "with_cycle_proxies": features,
    }

    all_folds: list[FoldResult] = []
    pooled: dict[str, Any] = {}
    for variant, variant_features in variants.items():
        folds, result = _evaluate_configuration(frame, target, variant_features, variant)
        all_folds.extend(folds)
        pooled[variant] = result

    positive_cases = int((frame[target] == 1).sum())
    return {
        "dataset": str(dataset),
        "target": target,
        "cohort_filter": cohort_filter,
        "validation_design": "leave-one-survey-cycle-out",
        "preprocessing": "training-cycle medians only",
        "usable_positive_cases": positive_cases,
        "power_status": "underpowered" if positive_cases < 20 else "exploratory",
        "variants": {name: values for name, values in variants.items()},
        "folds": [asdict(fold) for fold in all_folds],
        "pooled_out_of_cycle": pooled,
        "limitations": [
            "Cycles are repeated cross-sections, not longitudinal patient follow-up.",
            "The same feature-selection rules are reused across cycles.",
            "Cluster-bootstrap intervals resample ten cycles and remain imprecise.",
            "Self-reported pancreatic-cancer labels may be misclassified.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MetaboGuard Cycle-Held-Out Temporal Validation",
        "",
        "## Purpose",
        "",
        "Each NHANES survey cycle is held out in turn, with imputation and model fitting "
        "performed only on the remaining cycles. This is an internal-external temporal "
        "validation design: it evaluates transportability across time and exposes "
        "heterogeneity that a random split can conceal. Clinical prediction guidance "
        "recommends evaluating performance in new settings and examining heterogeneity "
        "rather than treating one internal validation as definitive "
        "([BMJ evaluation guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772854/); "
        "[Nieboer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5708595/)).",
        "",
        "## Feature variants",
        "",
        "- **clinical_only:** excludes `survey_cycle_index` and "
        "`hba1c_cycle_age_sex_z` to test transportability using patient-level clinical "
        "and metabolic features.",
        "- **with_cycle_proxies:** includes the complete feature set to quantify whether "
        "cycle-derived context helps or signals temporal dependence.",
        "",
        "## Pooled out-of-cycle results",
        "",
        "| Variant | Model | Rows | Positives | AUROC (cycle-bootstrap 95% CI) | "
        "AUPRC (cycle-bootstrap 95% CI) | AUPRC lift | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if report.get("power_status") == "underpowered":
        lines[3:3] = [
            "",
            "> **Do not use as a model-performance claim.** The corrected target "
            f"contains only {report.get('usable_positive_cases', 0)} usable positive "
            "cases. Results below are diagnostic pipeline checks only.",
        ]
    for variant, models in report["pooled_out_of_cycle"].items():
        for model, metric in models.items():
            lines.append(
                f"| {variant} | {model} | {metric['rows']:,} | {metric['positives']} | "
                f"{metric['auroc']:.3f} ({metric['auroc_ci_95'][0]:.3f}-"
                f"{metric['auroc_ci_95'][1]:.3f}) | "
                f"{metric['auprc']:.3f} ({metric['auprc_ci_95'][0]:.3f}-"
                f"{metric['auprc_ci_95'][1]:.3f}) | "
                f"{metric['auprc_lift']:.2f}x | {metric['brier_score']:.4f} |"
            )

    lines.extend([
        "",
        "## Per-cycle results",
        "",
        "| Variant | Model | Held-out cycle | Test rows | Positives | AUROC | AUPRC | Lift |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for fold in report["folds"]:
        lines.append(
            f"| {fold['variant']} | {fold['model']} | {fold['held_out_cycle']} | "
            f"{fold['test_rows']:,} | {fold['test_positives']} | "
            f"{fold['auroc']:.3f} | {fold['auprc']:.3f} | {fold['auprc_lift']:.2f}x |"
        )

    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Prefer the clinical-only variant if performance is similar; it has less "
        "dependence on survey-era context.",
        "- Large variation between held-out cycles indicates dataset shift and argues "
        "against deploying a frozen model without recalibration or refitting. Temporal "
        "studies show model stability can change even when average discrimination appears "
        "acceptable ([Lopes et al. 2023](https://doi.org/10.1016/j.heliyon.2023.e17139)).",
        "- These results remain exploratory because pancreatic-cancer events are rare and "
        "the outcome is self-reported prevalent disease, not future incident cancer.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd api",
        "python validate_cycle_holdout.py \\",
        "  --dataset ../data/nhanes_multicycle_v2.csv \\",
        "  --target PancreaticCancer \\",
        "  --cohort-filter diabetics_only",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="../data/nhanes_multicycle_v2.csv")
    parser.add_argument("--target", default="PancreaticCancer")
    parser.add_argument(
        "--cohort-filter",
        default="diabetics_only",
        choices=["diabetics_only", "none"],
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    cohort_filter = None if args.cohort_filter == "none" else args.cohort_filter
    report = validate(dataset, target=args.target, cohort_filter=cohort_filter)

    project_root = Path(__file__).resolve().parent.parent
    json_path = (
        Path(args.json_output).resolve()
        if args.json_output
        else project_root / "data" / "cycle_holdout_validation_v2.json"
    )
    markdown_path = (
        Path(args.markdown_output).resolve()
        if args.markdown_output
        else project_root / "docs" / "CYCLE_HOLDOUT_VALIDATION_V2.md"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(_markdown(report))
    print(json.dumps(report["pooled_out_of_cycle"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
