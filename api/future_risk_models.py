"""Simulation-only future-risk models, calibration and evaluation for MetaboGuard.

Contents:

* **Transparent baselines first**: horizon logistic regression, a discrete-time hazard model
  (cause-specific, with death as a competing event), and a gradient-boosted tree baseline.
  Interpretable survival/horizon baselines are the appropriate reference point for this kind
  of task (for example the eight-cancer UK Biobank Cox models,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10919929/, DOI 10.1093/jncics/pkae008).
* **Compact temporal neural model**: a masked GRU over irregular visit sequences with
  explicit time deltas, trained in PyTorch. Sequential-EHR designs of this shape are what
  recent work uses for horizon-specific cancer risk (design context only, not a comparator:
  https://linkinghub.elsevier.com/retrieve/pii/S266637912500432X, DOI 10.1016/j.xcrm.2025.102359;
  temporal diabetes prediction: https://pmc.ncbi.nlm.nih.gov/articles/PMC10599553/,
  DOI 10.1371/journal.pdig.0000354).
* **Calibration** on the validation split (isotonic with a Platt fallback), storing raw and
  calibrated outputs, because external validations repeatedly find that uncalibrated risk
  models over-predict (https://pmc.ncbi.nlm.nih.gov/articles/PMC3445426/, DOI 10.1136/bmj.e5900).
* **Evaluation** with eligibility masks: horizon AUROC/AP, Brier, calibration
  intercept/slope/curve, Harrell C-index, decision-curve scaffolding, false-alert burden,
  bootstrap CIs, subgroup checks and negative controls. Competing-risk validation follows the
  reporting logic of https://www.bmj.com/lookup/doi/10.1136/bmj-2021-069249 (DOI 10.1136/bmj-2021-069249).

Everything here is **simulation only**. Synthetic metrics verify the software and the
protocol; they say nothing about real-world calibration, clinical utility or early detection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from longitudinal_schema import (
    CapabilityState,
    HORIZON_DAYS,
    HORIZON_LABELS,
    MIN_EVENTS_PER_HORIZON,
    PREVENTION_SAFE_FEATURES,
    SUPPORTED_OUTCOMES,
    assert_outcome_allowed,
    assert_simulated_future_risk_allowed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_VERSION = "metaboguard-future-risk-v1"
SIMULATION_ONLY_BANNER = (
    "SIMULATION ONLY. Trained and evaluated on synthetic longitudinal data. Not validated "
    "for patient risk, not calibrated to any real population, and not evidence of early "
    "detection or clinical utility."
)


@dataclass
class FutureRiskConfig:
    outcomes: tuple[str, ...] = SUPPORTED_OUTCOMES
    horizons_days: tuple[int, ...] = HORIZON_DAYS
    seed: int = 20260805
    #: Temporal model.
    max_visits: int = 8
    hidden_dim: int = 48
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 6
    device: str = "cpu"
    #: Discrete-time hazard interval width in days.
    hazard_interval_days: int = 183
    bootstrap_rounds: int = 200
    alert_thresholds: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    smoke: bool = False
    #: A temporal model is only admissible if reversing visit order degrades AUROC by at least
    #: this much. Below it, the model is not demonstrably using time and is marked experimental
    #: and rejected rather than being presented as a temporal advance.
    time_reversal_min_auroc_drop: float = 0.02
    #: Usability floor for selection. A non-positive calibration slope means the model's ranking
    #: is unusable; near-chance discrimination cannot support an estimate. Failing either floor
    #: makes the horizon abstain.
    min_calibration_slope: float = 0.2
    min_auroc: float = 0.6

    def as_dict(self) -> dict[str, Any]:
        payload = {key: getattr(self, key) for key in self.__dataclass_fields__}
        for key in ("outcomes", "horizons_days", "alert_thresholds"):
            payload[key] = list(payload[key])
        return payload


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

BASELINE_FEATURE_SUFFIXES = ("_last", "_mean", "_slope_per_year", "_delta", "_observed_count")
CONTEXT_FEATURES = (
    "age_at_index_proxy",
    "visit_count",
    "visit_density_per_year",
    "history_days",
    "median_visit_gap_days",
    "missingness_burden",
)


def baseline_feature_columns(features: pd.DataFrame) -> list[str]:
    """Prevention-safe tabular columns only: no outcome, label, date or site column."""
    columns: list[str] = []
    for feature in PREVENTION_SAFE_FEATURES:
        for suffix in BASELINE_FEATURE_SUFFIXES:
            name = f"{feature}{suffix}"
            if name in features.columns:
                columns.append(name)
    for name in CONTEXT_FEATURES:
        if name in features.columns:
            columns.append(name)
    forbidden = ("label", "eligible", "_date", "_days", "cause", "outcome", "cancer_site", "censor")
    safe = [
        column
        for column in columns
        if not any(token in column.lower() for token in forbidden)
        or column in {"history_days", "median_visit_gap_days"}
    ]
    return safe


def _matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return features[columns].astype(float).to_numpy()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _make_tree(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=200,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            )
        ]
    )


def fit_horizon_baselines(
    features: pd.DataFrame,
    columns: list[str],
    outcome: str,
    horizon: int,
    splits: dict[str, list[str]],
    config: FutureRiskConfig,
) -> dict[str, Any]:
    """Fit logistic and tree baselines for one outcome/horizon on eligible patients only."""
    suffix = HORIZON_LABELS[horizon]
    label_column = f"{outcome}_{suffix}_label"
    mask_column = f"{outcome}_{suffix}_eligible"

    def subset(split: str) -> pd.DataFrame:
        frame = features[features["patient_id"].isin(splits[split])]
        return frame[frame[mask_column] == 1]

    train, validation = subset("train"), subset("validation")
    if train[label_column].sum() < 5:
        return {"status": "insufficient_events", "train_events": int(train[label_column].sum())}

    fitted: dict[str, Any] = {"status": "fitted", "models": {}}
    for name, estimator in (
        ("horizon_logistic", _make_logistic()),
        ("gradient_boosted_trees", _make_tree(config.seed)),
    ):
        estimator.fit(_matrix(train, columns), train[label_column].to_numpy())
        fitted["models"][name] = estimator
    fitted["train_rows"] = int(len(train))
    fitted["validation_rows"] = int(len(validation))
    return fitted


# ---------------------------------------------------------------------------
# Discrete-time cause-specific hazard model
# ---------------------------------------------------------------------------


def build_person_interval_frame(
    features: pd.DataFrame, outcome: str, config: FutureRiskConfig
) -> pd.DataFrame:
    """Expand each patient into discrete intervals with a cause-specific event indicator.

    Cause coding: 1 = target outcome, 2 = competing death, 0 = censored. Only intervals a
    patient actually entered are emitted, so censoring is handled by construction rather than
    by treating censored patients as negatives.
    """
    interval = config.hazard_interval_days
    horizon_max = max(config.horizons_days)
    rows: list[dict[str, Any]] = []
    time_column, cause_column = f"{outcome}_time_days", f"{outcome}_cause"
    for record in features.itertuples(index=False):
        time_to = float(getattr(record, time_column))
        cause = int(getattr(record, cause_column))
        intervals = int(np.ceil(min(max(time_to, 1.0), horizon_max) / interval))
        for index in range(intervals):
            start = index * interval
            end = start + interval
            entered = time_to > start
            if not entered:
                break
            event = int(cause == 1 and time_to <= end)
            competing = int(cause == 2 and time_to <= end)
            rows.append(
                {
                    "patient_id": record.patient_id,
                    "interval_index": index,
                    "interval_start_days": start,
                    "event": event,
                    "competing_event": competing,
                }
            )
            if event or competing or time_to <= end:
                break
    return pd.DataFrame(rows)


def fit_discrete_time_hazard(
    features: pd.DataFrame,
    columns: list[str],
    outcome: str,
    splits: dict[str, list[str]],
    config: FutureRiskConfig,
) -> dict[str, Any]:
    """Cause-specific discrete-time hazard: logistic on (features, interval index)."""
    intervals = build_person_interval_frame(features, outcome, config)
    joined = intervals.merge(features[["patient_id", *columns]], on="patient_id", how="left")
    train = joined[joined["patient_id"].isin(splits["train"])]
    if train["event"].sum() < 5:
        return {"status": "insufficient_events", "train_events": int(train["event"].sum())}
    design_columns = [*columns, "interval_index"]
    model = _make_logistic()
    model.fit(train[design_columns].astype(float).to_numpy(), train["event"].to_numpy())
    competing_model = None
    if train["competing_event"].sum() >= 5:
        competing_model = _make_logistic()
        competing_model.fit(
            train[design_columns].astype(float).to_numpy(), train["competing_event"].to_numpy()
        )
    return {
        "status": "fitted",
        "model": model,
        "competing_model": competing_model,
        "design_columns": design_columns,
        "interval_days": config.hazard_interval_days,
        "train_person_intervals": int(len(train)),
        "train_events": int(train["event"].sum()),
        "competing_handled": competing_model is not None,
    }


def hazard_cumulative_incidence(
    fitted: dict[str, Any], features: pd.DataFrame, horizon: int
) -> np.ndarray:
    """Cause-specific cumulative incidence to ``horizon``, accounting for competing death."""
    interval = fitted["interval_days"]
    steps = int(np.ceil(horizon / interval))
    columns = fitted["design_columns"][:-1]
    base = features[columns].astype(float).to_numpy()
    survival = np.ones(len(features))
    incidence = np.zeros(len(features))
    for index in range(steps):
        design = np.column_stack([base, np.full(len(features), index, dtype=float)])
        target_hazard = fitted["model"].predict_proba(design)[:, 1]
        competing_hazard = (
            fitted["competing_model"].predict_proba(design)[:, 1]
            if fitted.get("competing_model") is not None
            else np.zeros(len(features))
        )
        incidence += survival * target_hazard
        survival = survival * (1.0 - target_hazard - competing_hazard).clip(1e-9, 1.0)
    return np.clip(incidence, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Temporal neural model
# ---------------------------------------------------------------------------


def build_sequences(
    visit_matrix: pd.DataFrame,
    patient_ids: list[str],
    config: FutureRiskConfig,
    features: tuple[str, ...] = PREVENTION_SAFE_FEATURES,
    normaliser: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, tuple[float, float]]]:
    """Pad visit sequences into (patients, visits, features) with masks and time deltas."""
    feature_columns = [f"feature_{name}" for name in features]
    mask_columns = [f"mask_{name}" for name in features]
    frame = visit_matrix[visit_matrix["patient_id"].isin(patient_ids)]
    if normaliser is None:
        normaliser = {}
        for column in feature_columns:
            values = frame[column].astype(float)
            centre = float(values.median()) if values.notna().any() else 0.0
            spread = float(values.std()) if values.notna().any() and values.std() > 0 else 1.0
            normaliser[column] = (centre, spread)

    ordered_ids = [pid for pid in patient_ids if pid in set(frame["patient_id"])]
    sequences = np.zeros((len(ordered_ids), config.max_visits, len(features) * 2 + 1), dtype=np.float32)
    lengths = np.zeros(len(ordered_ids), dtype=np.int64)
    for row_index, patient_id in enumerate(ordered_ids):
        group = frame[frame["patient_id"] == patient_id].sort_values("visit_index").tail(config.max_visits)
        lengths[row_index] = len(group)
        for step, (_, visit) in enumerate(group.iterrows()):
            values = []
            for column in feature_columns:
                centre, spread = normaliser[column]
                raw = visit[column]
                values.append(0.0 if pd.isna(raw) else float((raw - centre) / spread))
            masks = [float(visit[column]) for column in mask_columns]
            delta = float(visit["delta_days_since_previous_visit"]) / 365.25
            sequences[row_index, step, :] = np.array([*values, *masks, delta], dtype=np.float32)
    padding_mask = (np.arange(config.max_visits)[None, :] < lengths[:, None]).astype(np.float32)
    return sequences, padding_mask, lengths, ordered_ids, normaliser



def build_temporal_module(input_dim: int, hidden_dim: int, horizons: int):
    """Construct the temporal architecture. Module-level so a saved artifact can rebuild it.

    Reads the **last valid hidden state** rather than a masked mean over GRU outputs: mean
    pooling is close to order-invariant and the first version of this model duly failed the
    time-reversal negative control, scoring identically on reversed visit sequences.
    """
    import torch
    from torch import nn

    class TemporalRiskModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim // 2, horizons),
            )

        def forward(self, sequence, padding):
            projected = torch.tanh(self.input_projection(sequence))
            outputs, _ = self.gru(projected)
            lengths = padding.sum(dim=1).clamp(min=1.0).long() - 1
            index = lengths.view(-1, 1, 1).expand(-1, 1, outputs.size(2))
            last = outputs.gather(1, index).squeeze(1)
            return self.head(last)

    return TemporalRiskModel()


def reload_temporal_model(state_dict: dict, input_dim: int, hidden_dim: int, horizons: int):
    """Rebuild a trained temporal model from a saved state dict for scoring-parity checks."""
    import torch

    model = build_temporal_module(input_dim, hidden_dim, horizons)
    model.load_state_dict({key: torch.tensor(value) for key, value in state_dict.items()})
    model.eval()
    return model


def train_temporal_model(
    visit_matrix: pd.DataFrame,
    features: pd.DataFrame,
    outcome: str,
    splits: dict[str, list[str]],
    config: FutureRiskConfig,
) -> dict[str, Any]:
    """Masked GRU over irregular visits with one head per horizon (multi-horizon output).

    A separate encoder from the cross-sectional MetaboGuard SSL model is used deliberately:
    that encoder consumes a single NHANES cross-section of 25 features fit on one-visit-per
    participant data, and its preprocessing and deviation reference are frozen to that
    distribution. Mapping it onto irregular multi-visit sequences with time deltas and masks
    is not defensible, so the temporal model learns its own representation and the SSL
    artifact stays untouched.
    """
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    horizon_columns = [
        (f"{outcome}_{HORIZON_LABELS[h]}_label", f"{outcome}_{HORIZON_LABELS[h]}_eligible", h)
        for h in config.horizons_days
    ]

    def tensors(split: str, normaliser=None):
        ids = [pid for pid in splits[split] if pid in set(features["patient_id"])]
        sequences, padding, _, ordered, norm = build_sequences(
            visit_matrix, ids, config, normaliser=normaliser
        )
        frame = features.set_index("patient_id").loc[ordered]
        labels = np.stack([frame[label].to_numpy() for label, _, _ in horizon_columns], axis=1)
        masks = np.stack([frame[mask].to_numpy() for _, mask, _ in horizon_columns], axis=1)
        return (
            torch.tensor(sequences, device=device),
            torch.tensor(padding, device=device),
            torch.tensor(labels, dtype=torch.float32, device=device),
            torch.tensor(masks, dtype=torch.float32, device=device),
            ordered,
            norm,
        )

    train_x, train_pad, train_y, train_m, _, normaliser = tensors("train")
    validation_x, validation_pad, validation_y, validation_m, _, _ = tensors("validation", normaliser)

    input_dim = train_x.shape[2]

    model = build_temporal_module(input_dim, config.hidden_dim, len(horizon_columns)).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(config.seed)

    history: list[dict[str, float]] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    stale = 0
    epochs = 3 if config.smoke else config.epochs
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        batch_losses = []
        for start in range(0, len(train_x), config.batch_size):
            index = permutation[start : start + config.batch_size]
            logits = model(train_x[index], train_pad[index])
            # Masked loss: ineligible (censored-before-horizon) patients contribute nothing.
            raw = loss_function(logits, train_y[index])
            masked = (raw * train_m[index]).sum() / train_m[index].sum().clamp(min=1.0)
            optimiser.zero_grad()
            masked.backward()
            optimiser.step()
            batch_losses.append(float(masked.detach()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x, validation_pad)
            raw = loss_function(validation_logits, validation_y)
            validation_loss = float(
                (raw * validation_m).sum() / validation_m.sum().clamp(min=1.0)
            )
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, stale = validation_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(patient_ids: list[str]) -> tuple[list[str], np.ndarray]:
        sequences, padding, _, ordered, _ = build_sequences(
            visit_matrix, patient_ids, config, normaliser=normaliser
        )
        if not ordered:
            return [], np.zeros((0, len(horizon_columns)))
        with torch.no_grad():
            logits = model(
                torch.tensor(sequences, device=device), torch.tensor(padding, device=device)
            )
            return ordered, torch.sigmoid(logits).cpu().numpy()

    return {
        "status": "fitted",
        "model": model,
        "state_dict": {k: v.numpy() for k, v in best_state.items()},
        "normaliser": normaliser,
        "history": history,
        "predict": predict,
        "input_dim": int(input_dim),
        "hidden_dim": int(config.hidden_dim),
        "max_visits": int(config.max_visits),
        "horizons": [h for _, _, h in horizon_columns],
        "architecture": (
            f"Linear({input_dim}->{config.hidden_dim}) + tanh -> GRU({config.hidden_dim}) -> "
            f"last valid hidden state -> MLP({config.hidden_dim}->{config.hidden_dim // 2}->"
            f"{len(horizon_columns)}) with per-horizon sigmoid"
        ),
        "separate_encoder_rationale": (
            "The cross-sectional MetaboGuard SSL encoder is frozen to a single-visit NHANES "
            "feature distribution; reusing it for irregular multi-visit sequences with time "
            "deltas and masks is not defensible, so this model learns its own representation."
        ),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def fit_calibrator(
    probabilities: np.ndarray, labels: np.ndarray, seed: int = 0
) -> dict[str, Any]:
    """Isotonic calibration with a Platt (logistic) fallback for small samples."""
    valid = np.isfinite(probabilities)
    probabilities, labels = probabilities[valid], labels[valid]
    if len(labels) < 50 or labels.sum() < 10 or labels.sum() == len(labels):
        model = LogisticRegression(max_iter=1000)
        model.fit(probabilities.reshape(-1, 1), labels)
        return {"method": "platt_logistic", "model": model, "n": int(len(labels))}
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(probabilities, labels)
    del seed
    return {"method": "isotonic", "model": isotonic, "n": int(len(labels))}


def apply_calibrator(calibrator: dict[str, Any], probabilities: np.ndarray) -> np.ndarray:
    if calibrator["method"] == "isotonic":
        return np.clip(calibrator["model"].predict(probabilities), 0.0, 1.0)
    return np.clip(calibrator["model"].predict_proba(probabilities.reshape(-1, 1))[:, 1], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def calibration_intercept_slope(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Calibration-in-the-large (intercept) and calibration slope on the logit scale."""
    epsilon = 1e-6
    logit = np.log(np.clip(probabilities, epsilon, 1 - epsilon) / (1 - np.clip(probabilities, epsilon, 1 - epsilon)))
    if len(np.unique(labels)) < 2:
        return {"intercept": None, "slope": None}
    slope_model = LogisticRegression(max_iter=1000)
    slope_model.fit(logit.reshape(-1, 1), labels)
    slope = float(slope_model.coef_[0][0])
    offset_model = LogisticRegression(max_iter=1000, fit_intercept=True)
    offset_model.fit(np.zeros((len(logit), 1)), labels)
    intercept = float(np.mean(labels) - np.mean(probabilities))
    return {"intercept_probability_difference": round(intercept, 6), "slope": round(slope, 4)}


def calibration_curve_points(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(probabilities)
    chunks = np.array_split(order, min(bins, max(1, len(order) // 10)))
    points = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        points.append(
            {
                "n": int(len(chunk)),
                "mean_predicted": round(float(probabilities[chunk].mean()), 6),
                "observed_rate": round(float(labels[chunk].mean()), 6),
            }
        )
    return points


def harrell_c_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray, sample: int = 4000, seed: int = 0) -> float | None:
    """Harrell's C for cause-specific risk; subsampled for tractability, seeded."""
    random = np.random.default_rng(seed)
    n = len(times)
    if n < 10:
        return None
    concordant = permissible = 0
    pairs = min(sample * 20, n * (n - 1) // 2)
    for _ in range(pairs):
        i, j = random.integers(0, n, 2)
        if i == j:
            continue
        if events[i] == 1 and (times[i] < times[j]):
            permissible += 1
            concordant += int(risk[i] > risk[j]) + 0.5 * int(risk[i] == risk[j])
        elif events[j] == 1 and (times[j] < times[i]):
            permissible += 1
            concordant += int(risk[j] > risk[i]) + 0.5 * int(risk[i] == risk[j])
    return round(float(concordant / permissible), 4) if permissible else None


def decision_curve(probabilities: np.ndarray, labels: np.ndarray, thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    """Net-benefit scaffolding plus false-alert burden at each alert threshold."""
    n = len(labels)
    prevalence = float(labels.mean()) if n else 0.0
    rows = []
    for threshold in thresholds:
        flagged = probabilities >= threshold
        true_positive = int((flagged & (labels == 1)).sum())
        false_positive = int((flagged & (labels == 0)).sum())
        weight = threshold / (1 - threshold) if threshold < 1 else np.inf
        net_benefit = (true_positive / n) - (false_positive / n) * weight if n else None
        rows.append(
            {
                "threshold": threshold,
                "flagged": int(flagged.sum()),
                "flagged_fraction": round(float(flagged.mean()), 6) if n else None,
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_alerts_per_100_screened": round(false_positive / n * 100, 3) if n else None,
                "net_benefit_model": round(float(net_benefit), 6) if net_benefit is not None else None,
                "net_benefit_treat_all": round(prevalence - (1 - prevalence) * weight, 6)
                if threshold < 1
                else None,
            }
        )
    return rows


def bootstrap_ci(
    probabilities: np.ndarray, labels: np.ndarray, rounds: int, seed: int
) -> dict[str, Any]:
    random = np.random.default_rng(seed)
    auroc, average_precision = [], []
    for _ in range(rounds):
        index = random.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        auroc.append(roc_auc_score(labels[index], probabilities[index]))
        average_precision.append(average_precision_score(labels[index], probabilities[index]))
    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]
    return {"auroc_95ci": interval(auroc), "average_precision_95ci": interval(average_precision), "rounds": len(auroc)}


def evaluate_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    outcome: str,
    horizon: int,
    config: FutureRiskConfig,
    subgroup_columns: tuple[str, ...] = ("age_group", "sex_code", "missingness_group", "visit_density_group"),
) -> dict[str, Any]:
    """Horizon metrics on eligible patients only, with subgroups and CIs."""
    suffix = HORIZON_LABELS[horizon]
    labels = frame[f"{outcome}_{suffix}_label"].to_numpy().astype(int)
    if len(np.unique(labels)) < 2:
        return {"status": "not_evaluable", "reason": "single class among eligible patients", "n": int(len(labels))}
    result: dict[str, Any] = {
        "status": "evaluated",
        "n_eligible": int(len(labels)),
        "events": int(labels.sum()),
        "non_events": int((labels == 0).sum()),
        "prevalence": round(float(labels.mean()), 6),
        "auroc": round(float(roc_auc_score(labels, probabilities)), 4),
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4),
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "calibration": calibration_intercept_slope(probabilities, labels),
        "calibration_curve": calibration_curve_points(probabilities, labels),
        "decision_curve": decision_curve(probabilities, labels, config.alert_thresholds),
        **bootstrap_ci(probabilities, labels, config.bootstrap_rounds if not config.smoke else 40, config.seed),
    }
    if f"{outcome}_time_days" in frame.columns:
        result["c_index_cause_specific"] = harrell_c_index(
            frame[f"{outcome}_time_days"].to_numpy(),
            (frame[f"{outcome}_cause"].to_numpy() == 1).astype(int),
            probabilities,
            seed=config.seed,
        )
    subgroups: dict[str, Any] = {}
    for column in subgroup_columns:
        if column not in frame.columns:
            continue
        per_group = {}
        for value, group in frame.groupby(column):
            group_labels = group[f"{outcome}_{suffix}_label"].to_numpy().astype(int)
            group_probabilities = probabilities[frame[column].to_numpy() == value]
            if len(np.unique(group_labels)) < 2 or group_labels.sum() < 10:
                per_group[str(value)] = {"status": "suppressed", "n": int(len(group_labels)), "events": int(group_labels.sum())}
                continue
            per_group[str(value)] = {
                "status": "evaluated",
                "n": int(len(group_labels)),
                "events": int(group_labels.sum()),
                "auroc": round(float(roc_auc_score(group_labels, group_probabilities)), 4),
                "brier": round(float(brier_score_loss(group_labels, group_probabilities)), 6),
            }
        subgroups[column] = per_group
    result["subgroups"] = subgroups
    return result


def add_subgroup_columns(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    age_source = "DEMO_RIDAGEYR_last" if "DEMO_RIDAGEYR_last" in frame.columns else None
    if age_source:
        frame["age_group"] = pd.cut(
            frame[age_source], [0, 45, 60, 75, 130], labels=["<45", "45-59", "60-74", "75+"]
        ).astype(str)
    if "sex_code" not in frame.columns:
        frame["sex_code"] = "unknown"
    frame["missingness_group"] = pd.qcut(
        frame["missingness_burden"].rank(method="first"), 3, labels=["low", "medium", "high"]
    ).astype(str)
    frame["visit_density_group"] = pd.qcut(
        frame["visit_density_per_year"].rank(method="first"), 3, labels=["sparse", "medium", "dense"]
    ).astype(str)
    return frame