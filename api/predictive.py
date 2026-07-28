"""
Predictive Baseline Module
==========================
Provides a lightweight prediction baseline for Diabetes and Cancer risk using
NHANES features. This module is intentionally dependency-light (NumPy/Pandas only)
and reports imbalance-aware metrics (AUROC/AUPRC/recall/F1/balanced accuracy)
instead of relying on raw accuracy alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_COLUMNS = {"Diabetes", "Cancer", "Obesity"}


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.where(numeric == 1, 1.0, np.where(numeric == 2, 0.0, np.nan))


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "Obesity" not in prepared.columns and "BMX_BMXBMI" in prepared.columns:
        bmi = pd.to_numeric(prepared["BMX_BMXBMI"], errors="coerce")
        prepared["Obesity"] = np.where(bmi >= 30, 1.0, np.where(bmi.notna(), 0.0, np.nan))

    if "Diabetes" not in prepared.columns and "DIQ_DIQ010" in prepared.columns:
        prepared["Diabetes"] = _coerce_binary(prepared["DIQ_DIQ010"])

    if "Cancer" not in prepared.columns and "MCQ_MCQ220" in prepared.columns:
        prepared["Cancer"] = _coerce_binary(prepared["MCQ_MCQ220"])

    # Cancer is the only strictly-required column. Diabetes/Obesity are best-effort:
    # they may be entirely absent on datasets such as TCGA-CDR.
    if "Cancer" not in prepared.columns:
        raise ValueError("Dataset is missing required column: Cancer")

    return prepared


def _stratified_split(y: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    if len(positive_idx) < 2 or len(negative_idx) < 2:
        raise ValueError("Need at least 2 positive and 2 negative rows for train/test evaluation.")

    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    test_pos = max(1, int(round(len(positive_idx) * test_fraction)))
    test_neg = max(1, int(round(len(negative_idx) * test_fraction)))

    test_idx = np.concatenate([positive_idx[:test_pos], negative_idx[:test_neg]])
    train_idx = np.concatenate([positive_idx[test_pos:], negative_idx[test_neg:]])

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/test split failed due to insufficient samples.")

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    iterations: int = 1600,
    learning_rate: float = 0.08,
    l2: float = 0.002,
) -> np.ndarray:
    weights = np.zeros(x_train.shape[1], dtype=float)

    for _ in range(iterations):
        predictions = _sigmoid(x_train @ weights)
        error = predictions - y_train
        gradient = (x_train.T @ error) / len(y_train)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient

    return weights


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    n_pos = int(positive_mask.sum())
    n_neg = int(negative_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

    rank_sum_pos = ranks[positive_mask].sum()
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate(([n_pos / len(y_true)], precision))
    recall = np.concatenate(([0.0], recall))

    area = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(area)


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = _safe_divide(tp + tn, len(y_true))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy),
        "f1": f1,
        "auroc": _roc_auc(y_true, y_prob),
        "auprc": _pr_auc(y_true, y_prob),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "positive_rate": float(y_true.mean()),
    }


def _candidate_features_for_target(target: str, columns: set[str]) -> list[str]:
    shared = [
        "Obesity",
        "DEMO_RIDAGEYR",
        "DEMO_RIAGENDR",
        "DEMO_RIDRETH3",
        "BMX_BMXBMI",
        "SMQ_SMQ020",
        "ALQ_ALQ101",
    ]

    # TCGA-specific features. tcga_followup_days/event and tcga_pfi_days/event
    # are deliberately excluded — the 5-year mortality and progression labels
    # are derived from them (data leakage).
    tcga_extras = [
        c for c in (
            "tcga_stage_ordinal",
            "tcga_grade_ordinal",
            "tcga_tumor_status",
            "tcga_treatment_response",
        )
        if c in columns
    ]
    tcga_type_flags = sorted(c for c in columns if c.startswith("tcga_type_"))

    if target == "Diabetes":
        return ["Cancer", *shared]

    if target == "Cancer":
        return ["Diabetes", *shared, *tcga_extras, *tcga_type_flags]

    return shared


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _build_target_model(df: pd.DataFrame, target: str) -> dict[str, object]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is unavailable.")

    candidate_features = [
        column
        for column in _candidate_features_for_target(target, set(df.columns))
        if column in df.columns and df[column].notna().any()
    ]
    candidate_features = [column for column in candidate_features if column != target]

    if not candidate_features:
        raise ValueError(f"No usable features found for target '{target}'.")

    model_df = df[[target, *candidate_features]].copy()
    model_df[target] = pd.to_numeric(model_df[target], errors="coerce")

    for feature in candidate_features:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")

    model_df = model_df.dropna(subset=[target, *candidate_features])
    model_df = model_df[(model_df[target] == 0) | (model_df[target] == 1)]

    if len(model_df) < 80:
        raise ValueError(f"Insufficient rows after cleaning for target '{target}'.")

    y = model_df[target].to_numpy(dtype=int)
    x = model_df[candidate_features].to_numpy(dtype=float)

    train_idx, test_idx = _stratified_split(y, test_fraction=0.2, seed=42)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    x_train_scaled = (x_train - means) / stds
    x_test_scaled = (x_test - means) / stds

    x_train_design = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled])
    x_test_design = np.column_stack([np.ones(len(x_test_scaled)), x_test_scaled])

    weights = _train_logistic_regression(x_train_design, y_train)
    probabilities = _sigmoid(x_test_design @ weights)

    model_metrics = _classification_metrics(y_test, probabilities, threshold=0.5)
    baseline_probability = float(y_train.mean())
    baseline_probs = np.full(len(y_test), baseline_probability, dtype=float)
    baseline_metrics = _classification_metrics(y_test, baseline_probs, threshold=0.5)

    return {
        "target": target,
        "features": candidate_features,
        "rows_used": int(len(model_df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "threshold": 0.5,
        "baseline_probability": round(baseline_probability, 6),
        "metrics": _round_metrics(model_metrics),
        "baseline_metrics": _round_metrics(baseline_metrics),
    }


def _try_build(prepared: pd.DataFrame, target: str) -> dict[str, object] | dict[str, str]:
    if target not in prepared.columns or not prepared[target].notna().any():
        return {"status": "skipped", "reason": f"target column '{target}' unavailable in this dataset"}
    try:
        return _build_target_model(prepared, target)
    except ValueError as error:
        return {"status": "skipped", "reason": str(error)}


def execute_predictive_baseline(data_path: str) -> dict[str, object]:
    df = pd.read_csv(data_path)
    prepared = _prepare_dataframe(df)

    diabetes_result = _try_build(prepared, "Diabetes")
    cancer_result = _try_build(prepared, "Cancer")

    return {
        "model": "logistic_regression_gradient_descent",
        "notes": [
            "Cross-sectional baseline with fixed 80/20 stratified split.",
            "Use AUROC/AUPRC/recall/balanced_accuracy for imbalanced-outcome evaluation.",
            "This is predictive risk modeling and does not establish causality.",
            "Targets are skipped when the dataset does not carry that label (e.g. Diabetes on TCGA-CDR).",
        ],
        "results": {
            "diabetes": diabetes_result,
            "cancer": cancer_result,
        },
    }
