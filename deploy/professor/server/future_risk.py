"""Simulation-only future-risk scoring on a portable, dependency-light runtime.

Why a portable artifact
-----------------------
The authoritative artifact
(``model_artifacts/future_risk/simulation_synthea_pooled/artifact/future_risk_models.joblib``)
holds fitted scikit-learn pipelines and a PyTorch state dict. Neither
scikit-learn/SciPy nor PyTorch can be installed in the Vercel Hobby function, so
``future_risk_export.py`` exports the **selected** models per horizon into a
portable NumPy/JSON representation that this module replays:

* ``discrete_time_hazard`` - ``SimpleImputer(median, add_indicator)`` ->
  ``StandardScaler`` -> ``LogisticRegression``, then the cause-specific
  cumulative-incidence recursion with the competing-risk hazard;
* ``gradient_boosted_trees`` - the ``HistGradientBoostingClassifier`` decision
  trees exported node by node and traversed in NumPy;
* ``temporal_gru`` - input projection, one GRU layer and the MLP head exported as
  weight matrices and evaluated in NumPy.

Nothing is approximated and no probability is invented. Parity against the
authoritative joblib/PyTorch artifact is measured by the export script on
representative synthetic histories and recorded in
``assets/future_risk/parity_report.json``; the tests assert it.

Contract
--------
* Simulation only. Every payload carries the artifact's own banner and
  ``simulation_only: true``.
* A horizon either returns a simulated estimate (raw + calibrated + flags) or an
  explicit abstention. Unsupported horizons abstain; they are never filled in
  from a different model.
* Type 1 diabetes and site-specific cancer stay disabled.
* At least two distinct visit times are required. Cross-sectional input is
  rejected, never scored.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

PORTABLE_DIR = config.ASSETS_DIR / "future_risk"
PORTABLE_JSON = PORTABLE_DIR / "portable_models.json"
PORTABLE_ARRAYS = PORTABLE_DIR / "portable_models.npz"
PARITY_REPORT = PORTABLE_DIR / "parity_report.json"
SUMMARY_JSON = PORTABLE_DIR / "artifact_summary.json"

#: Features the simulated history may carry (PREVENTION_SAFE_FEATURES upstream).
HISTORY_FEATURES: tuple[str, ...] = (
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

HORIZON_DAYS: dict[str, int] = {"1y": 365, "3y": 1095, "5y": 1825}
OUTCOMES: tuple[str, ...] = ("type2_diabetes", "pan_cancer")
DISABLED_OUTCOMES: dict[str, str] = {
    "type1_diabetes": (
        "Disabled: autoimmune onset is not modelled and the simulated cohort carries no "
        "usable type 1 incidence."
    ),
    "cancer_site_specific": (
        "Disabled: site-specific cancer requires site-labelled incident outcomes that the "
        "simulated cohort does not provide."
    ),
}

MIN_VISITS = 2

SIMULATION_BANNER_FALLBACK = (
    "SIMULATION ONLY. Trained and evaluated on synthetic longitudinal data. Not validated "
    "for patient risk, not calibrated to any real population, and not evidence of early "
    "detection or clinical utility."
)

CLINICAL_ENDPOINT_MESSAGE = (
    "Clinical future-risk scoring is not available. The only future-risk models in this "
    "deployment are simulation-only, trained on synthetic longitudinal data. Producing a "
    "patient-facing risk estimate from them is prohibited."
)


#: Keys a simulated visit row may carry. Anything else is refused outright.
ALLOWED_VISIT_KEYS: frozenset[str] = frozenset(
    {
        "visit_index",
        "days_before_index",
        "years_before_index",
        *HISTORY_FEATURES,
    }
)


def reject_identifier_fields(visits: Any) -> list[str]:
    """Return any submitted visit key outside the simulated-measurement allowlist.

    This deployment never accepts a patient identifier, a date of birth, a name, a
    record number or a free-text note on this route: a simulated history needs only
    a relative visit time and simulated measurements.
    """
    if not isinstance(visits, list):
        return []
    offending: list[str] = []
    for visit in visits:
        if not isinstance(visit, dict):
            continue
        for key in visit:
            if key not in ALLOWED_VISIT_KEYS and key not in offending:
                offending.append(str(key))
    return offending


class SimulationInputRejected(Exception):
    """Raised when a submitted history cannot be scored at all."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


# ---------------------------------------------------------------------------
# Portable artifact loading
# ---------------------------------------------------------------------------


def artifact_available() -> bool:
    return PORTABLE_JSON.exists() and PORTABLE_ARRAYS.exists()


@lru_cache(maxsize=1)
def portable_manifest() -> dict[str, Any]:
    if not PORTABLE_JSON.exists():
        raise FileNotFoundError(
            "Portable future-risk artifact is missing. Run prepare_assets.py."
        )
    return json.loads(PORTABLE_JSON.read_text())


@lru_cache(maxsize=1)
def portable_arrays() -> dict[str, np.ndarray]:
    archive = np.load(PORTABLE_ARRAYS)
    return {key: archive[key] for key in archive.files}


@lru_cache(maxsize=1)
def artifact_summary() -> dict[str, Any]:
    if not SUMMARY_JSON.exists():
        return {}
    return json.loads(SUMMARY_JSON.read_text())


@lru_cache(maxsize=1)
def parity_report() -> dict[str, Any]:
    if not PARITY_REPORT.exists():
        return {}
    return json.loads(PARITY_REPORT.read_text())


def _array(key: str) -> np.ndarray:
    arrays = portable_arrays()
    if key not in arrays:
        raise KeyError(f"portable artifact is missing array '{key}'")
    return arrays[key]


# ---------------------------------------------------------------------------
# History -> features
# ---------------------------------------------------------------------------


def _visit_rows_to_frame(visits: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, visit in enumerate(visits):
        days = visit.get("days_before_index")
        if days is None:
            raise SimulationInputRejected("Every visit needs days_before_index.")
        try:
            days_value = float(days)
        except (TypeError, ValueError):
            raise SimulationInputRejected("days_before_index must be numeric.")
        if days_value < 0:
            raise SimulationInputRejected("days_before_index must be zero or positive.")
        record: dict[str, Any] = {
            "visit_index": index,
            "days_before_index": days_value,
        }
        for feature in HISTORY_FEATURES:
            raw = visit.get(feature)
            if raw is None or raw == "":
                record[feature] = np.nan
                continue
            try:
                record[feature] = float(raw)
            except (TypeError, ValueError):
                raise SimulationInputRejected(
                    f"Feature {feature} must be numeric or blank."
                )
        rows.append(record)
    return pd.DataFrame(rows)


def validate_history(visits: Any) -> pd.DataFrame:
    """Reject anything that is not a multi-visit synthetic history."""
    if not isinstance(visits, list) or not visits:
        raise SimulationInputRejected("A visit timeline is required.")
    if len(visits) < MIN_VISITS:
        raise SimulationInputRejected(
            "Future-risk simulation needs at least two visits. A single visit is "
            "cross-sectional and cannot support a time horizon.",
            {"visits_supplied": len(visits), "visits_required": MIN_VISITS},
        )
    if len(visits) > 24:
        raise SimulationInputRejected("At most 24 visits are accepted.")
    frame = _visit_rows_to_frame(visits)
    distinct = frame["days_before_index"].nunique()
    if distinct < MIN_VISITS:
        raise SimulationInputRejected(
            "Visits must occur at two or more distinct times. Repeated measurements at one "
            "time are cross-sectional and cannot support a time horizon.",
            {"distinct_visit_times": int(distinct)},
        )
    observed = frame[list(HISTORY_FEATURES)].notna().sum(axis=1)
    if int((observed > 0).sum()) < MIN_VISITS:
        raise SimulationInputRejected(
            "At least two visits must carry at least one measured feature."
        )
    # Oldest visit first, as the upstream visit matrix expects.
    return frame.sort_values("days_before_index", ascending=False).reset_index(drop=True)


def build_feature_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(visit_matrix, tabular_features)`` for one simulated patient.

    Mirrors ``build_visit_matrix`` and ``build_patient_features`` upstream: the
    same relative times, gaps, density, masks, and the same ``_last``, ``_mean``,
    ``_slope_per_year``, ``_delta`` and ``_observed_count`` definitions. Parity of
    the resulting feature vector against the upstream builders is asserted by the
    export script and by ``tests/test_future_risk.py``.
    """
    patient_id = "simulated-history"
    visits = history.reset_index(drop=True)
    relative_days = -visits["days_before_index"].astype(float).to_numpy()
    span_days = max(float(relative_days.max() - relative_days.min()), 1.0)
    density = round(float(len(visits) / (span_days / 365.25)), 4)

    matrix_rows: list[dict[str, Any]] = []
    previous: float | None = None
    for index, relative in enumerate(relative_days):
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "visit_index": index,
            "relative_time_days": float(relative),
            "delta_days_since_previous_visit": (
                0.0 if previous is None else float(relative - previous)
            ),
            "visit_density_per_year": density,
        }
        observed_count = 0
        for feature in HISTORY_FEATURES:
            value = visits.loc[index, feature]
            present = bool(pd.notna(value))
            row[f"feature_{feature}"] = float(value) if present else np.nan
            row[f"mask_{feature}"] = int(present)
            observed_count += int(present)
        row["observed_feature_count"] = observed_count
        matrix_rows.append(row)
        previous = float(relative)
    matrix = pd.DataFrame(matrix_rows)

    group = matrix.sort_values("visit_index")
    features: dict[str, Any] = {"patient_id": patient_id}
    features["visit_count"] = int(len(group))
    features["visit_density_per_year"] = float(group["visit_density_per_year"].iloc[-1])
    features["history_days"] = float(-group["relative_time_days"].min())
    gaps = group["delta_days_since_previous_visit"].iloc[1:]
    features["median_visit_gap_days"] = float(gaps.median() or 0.0)
    features["missingness_burden"] = float(
        1.0 - group[[f"mask_{feature}" for feature in HISTORY_FEATURES]].to_numpy().mean()
    )
    for feature in HISTORY_FEATURES:
        values = group[f"feature_{feature}"].astype(float)
        times = group["relative_time_days"].astype(float) / 365.25
        observed = values.notna()
        features[f"{feature}_observed_count"] = int(observed.sum())
        if observed.sum() == 0:
            for suffix in ("_last", "_mean", "_slope_per_year", "_delta"):
                features[f"{feature}{suffix}"] = np.nan
            continue
        features[f"{feature}_last"] = float(values[observed].iloc[-1])
        features[f"{feature}_mean"] = float(values[observed].mean())
        features[f"{feature}_delta"] = float(
            values[observed].iloc[-1] - values[observed].iloc[0]
        )
        if observed.sum() >= 2 and times[observed].std() > 0:
            slope = np.polyfit(times[observed], values[observed], 1)[0]
            features[f"{feature}_slope_per_year"] = float(slope)
        else:
            features[f"{feature}_slope_per_year"] = 0.0
    return matrix, pd.DataFrame([features])


def design_matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    frame = features.reindex(columns=columns)
    return frame.astype(float).to_numpy()


def build_sequence(matrix: pd.DataFrame, normaliser: dict[str, list[float]], max_visits: int):
    """Pad the visit matrix into the GRU input, mirroring ``build_sequences`` upstream."""
    group = matrix.sort_values("visit_index").tail(max_visits)
    width = len(HISTORY_FEATURES) * 2 + 1
    sequence = np.zeros((1, max_visits, width), dtype=np.float32)
    for step, (_, visit) in enumerate(group.iterrows()):
        values = []
        for feature in HISTORY_FEATURES:
            centre, spread = normaliser[f"feature_{feature}"]
            raw = visit[f"feature_{feature}"]
            values.append(
                0.0 if pd.isna(raw) else float((float(raw) - centre) / spread)
            )
        masks = [float(visit[f"mask_{feature}"]) for feature in HISTORY_FEATURES]
        delta = float(visit["delta_days_since_previous_visit"]) / 365.25
        sequence[0, step, :] = np.array([*values, *masks, delta], dtype=np.float32)
    length = int(len(group))
    padding = (np.arange(max_visits)[None, :] < np.array([length])[:, None]).astype(np.float32)
    return sequence, padding, length


# ---------------------------------------------------------------------------
# Portable predictors
# ---------------------------------------------------------------------------


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def logistic_probability(spec: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    """SimpleImputer(median, add_indicator) -> StandardScaler -> LogisticRegression."""
    prefix = spec["array_prefix"]
    statistics = _array(f"{prefix}/imputer_statistics")
    indicator_columns = _array(f"{prefix}/indicator_columns").astype(int)
    mean = _array(f"{prefix}/scaler_mean")
    scale = _array(f"{prefix}/scaler_scale")
    coefficients = _array(f"{prefix}/coef")
    intercept = _array(f"{prefix}/intercept")

    values = matrix.astype(float).copy()
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(statistics, np.where(missing)[1])
    if indicator_columns.size:
        indicators = missing[:, indicator_columns].astype(float)
        values = np.hstack([values, indicators])
    scaled = (values - mean) / scale
    logits = scaled @ coefficients.reshape(-1) + float(intercept.reshape(-1)[0])
    return _sigmoid(logits)


def tree_ensemble_raw(spec: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    """Traverse exported HistGradientBoosting decision trees in NumPy."""
    prefix = spec["array_prefix"]
    left = _array(f"{prefix}/left").astype(np.int64)
    right = _array(f"{prefix}/right").astype(np.int64)
    feature = _array(f"{prefix}/feature").astype(np.int64)
    threshold = _array(f"{prefix}/threshold").astype(float)
    missing_left = _array(f"{prefix}/missing_go_to_left").astype(bool)
    is_leaf = _array(f"{prefix}/is_leaf").astype(bool)
    value = _array(f"{prefix}/value").astype(float)
    tree_start = _array(f"{prefix}/tree_start").astype(np.int64)
    baseline = float(spec["baseline_prediction"])

    values = matrix.astype(float)
    raw = np.full(values.shape[0], baseline, dtype=float)
    for row_index in range(values.shape[0]):
        row = values[row_index]
        total = baseline
        for start in tree_start:
            node = int(start)
            while not is_leaf[node]:
                column = int(feature[node])
                sample = row[column]
                if not np.isfinite(sample):
                    node = int(left[node]) if missing_left[node] else int(right[node])
                elif sample <= threshold[node]:
                    node = int(left[node])
                else:
                    node = int(right[node])
            total += value[node]
        raw[row_index] = total
    return raw


def tree_probability(spec: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    return _sigmoid(tree_ensemble_raw(spec, matrix))


def _gelu_tanh(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * values**3))
    )


def gru_probabilities(spec: dict[str, Any], sequence: np.ndarray, length: int) -> np.ndarray:
    """Input projection -> tanh -> GRU -> last valid hidden state -> MLP head -> sigmoid."""
    prefix = spec["array_prefix"]
    projection_weight = _array(f"{prefix}/input_projection_weight")
    projection_bias = _array(f"{prefix}/input_projection_bias")
    weight_ih = _array(f"{prefix}/gru_weight_ih")
    weight_hh = _array(f"{prefix}/gru_weight_hh")
    bias_ih = _array(f"{prefix}/gru_bias_ih")
    bias_hh = _array(f"{prefix}/gru_bias_hh")
    head0_weight = _array(f"{prefix}/head0_weight")
    head0_bias = _array(f"{prefix}/head0_bias")
    head2_weight = _array(f"{prefix}/head2_weight")
    head2_bias = _array(f"{prefix}/head2_bias")

    batch, steps, _ = sequence.shape
    hidden_dim = weight_hh.shape[1]
    projected = np.tanh(sequence @ projection_weight.T + projection_bias)

    hidden = np.zeros((batch, hidden_dim), dtype=np.float64)
    outputs = np.zeros((batch, steps, hidden_dim), dtype=np.float64)
    for step in range(steps):
        step_input = projected[:, step, :].astype(np.float64)
        gates_input = step_input @ weight_ih.T + bias_ih
        gates_hidden = hidden @ weight_hh.T + bias_hh
        reset = _sigmoid(gates_input[:, :hidden_dim] + gates_hidden[:, :hidden_dim])
        update = _sigmoid(
            gates_input[:, hidden_dim : 2 * hidden_dim]
            + gates_hidden[:, hidden_dim : 2 * hidden_dim]
        )
        candidate = np.tanh(
            gates_input[:, 2 * hidden_dim :] + reset * gates_hidden[:, 2 * hidden_dim :]
        )
        hidden = (1.0 - update) * candidate + update * hidden
        outputs[:, step, :] = hidden

    index = max(int(length) - 1, 0)
    last = outputs[:, index, :]
    hidden_layer = _gelu_tanh(last @ head0_weight.T + head0_bias)
    logits = hidden_layer @ head2_weight.T + head2_bias
    return _sigmoid(logits)[0]


def apply_calibrator(spec: dict[str, Any], probability: float) -> float:
    """Isotonic (linear interpolation over exported thresholds) or Platt logistic."""
    if spec["method"] == "isotonic":
        prefix = spec["array_prefix"]
        x_thresholds = _array(f"{prefix}/x_thresholds")
        y_thresholds = _array(f"{prefix}/y_thresholds")
        calibrated = float(
            np.interp(
                probability,
                x_thresholds,
                y_thresholds,
                left=float(y_thresholds[0]),
                right=float(y_thresholds[-1]),
            )
        )
        return float(np.clip(calibrated, 0.0, 1.0))
    coefficient = float(spec["coef"])
    intercept = float(spec["intercept"])
    return float(np.clip(_sigmoid(np.array([probability * coefficient + intercept]))[0], 0.0, 1.0))


def hazard_cumulative_incidence(
    spec: dict[str, Any], base_matrix: np.ndarray, horizon_days: int
) -> float:
    """Cause-specific cumulative incidence with the competing-risk hazard."""
    interval = int(spec["interval_days"])
    steps = int(np.ceil(horizon_days / interval))
    survival = 1.0
    incidence = 0.0
    for index in range(steps):
        design = np.column_stack([base_matrix, np.full(base_matrix.shape[0], index, dtype=float)])
        target = float(logistic_probability(spec["event"], design)[0])
        competing = (
            float(logistic_probability(spec["competing"], design)[0])
            if spec.get("competing")
            else 0.0
        )
        incidence += survival * target
        survival = survival * float(np.clip(1.0 - target - competing, 1e-9, 1.0))
    return float(np.clip(incidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Capability and scoring
# ---------------------------------------------------------------------------


def capability() -> dict[str, Any]:
    """Everything the UI needs to decide which rows can carry an estimate."""
    manifest = portable_manifest() if artifact_available() else {}
    summary = artifact_summary()
    supported = manifest.get("supported", {})
    abstained = manifest.get("abstained", {})
    return {
        "simulation_only": True,
        "clinical_use": "prohibited",
        "capability_state": "simulation_only_longitudinal",
        "banner": manifest.get("banner", SIMULATION_BANNER_FALLBACK),
        "data_source": "synthetic longitudinal cohort (Synthea-derived, pooled)",
        "requires_simulation_mode": True,
        "minimum_visits": MIN_VISITS,
        "history_features": list(HISTORY_FEATURES),
        "outcomes": [
            {"id": "type2_diabetes", "label": "Type 2 diabetes", "enabled": True},
            {"id": "pan_cancer", "label": "Cancer (pan-cancer composite)", "enabled": True},
        ],
        "disabled_outcomes": DISABLED_OUTCOMES,
        "horizons": list(HORIZON_DAYS),
        "supported_horizons": {
            key: {
                "selected_model": value["selected_model"],
                "calibration_method": value.get("calibration_method"),
                "metrics": value.get("metrics", {}),
            }
            for key, value in supported.items()
        },
        "abstained_horizons": abstained,
        # The panel upstream reads capability.artifact.selection to gate rows.
        "artifact": {
            "artifact_version": manifest.get("source_artifact", {}).get("artifact_version"),
            "code_version": manifest.get("source_artifact", {}).get("code_version"),
            "created_at": manifest.get("source_artifact", {}).get("created_at"),
            "portable_schema_version": manifest.get("schema_version"),
            "selection": {key: value.get("metrics", {}) for key, value in supported.items()},
            "package_versions": manifest.get("source_artifact", {}).get("package_versions"),
        },
        "parity": {
            key: value
            for key, value in parity_report().items()
            if key in {"max_abs_difference", "histories_compared", "verdict", "generated_at"}
        },
        "evaluation_caveats": summary.get("caveats", []),
        "competing_outcomes": summary.get("competing_outcomes"),
        "portable_artifact_available": artifact_available(),
        "prohibited_claims": [
            "This is a patient's risk of developing cancer or diabetes.",
            "These probabilities apply to a real population.",
            "The reported AUROC is clinical performance.",
            "A high simulated estimate warrants clinical action.",
        ],
    }


def _horizon_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(entry.get("metrics", {}))
    return {
        "auroc": metrics.get("auroc"),
        "auroc_ci": metrics.get("auroc_ci"),
        "brier": metrics.get("brier"),
        "calibration_slope": metrics.get("calibration_slope"),
        "calibration_intercept": metrics.get("calibration_intercept"),
        "test_split_events": metrics.get("test_split_events"),
        "test_split_n": metrics.get("test_split_n"),
        "underpowered_test_split": metrics.get("underpowered_test_split"),
        "wide_confidence_interval": metrics.get("wide_confidence_interval"),
    }


def score_history(visits: list[dict[str, Any]], simulation_mode: bool) -> dict[str, Any]:
    """Score one simulated history. Raises :class:`SimulationInputRejected` on bad input."""
    if simulation_mode is not True:
        raise SimulationInputRejected(
            "simulation_mode must be true. These models are simulation-only and cannot be "
            "used for clinical scoring."
        )
    if not artifact_available():
        raise SimulationInputRejected(
            "The portable future-risk artifact is not installed in this deployment."
        )
    history = validate_history(visits)
    matrix, features = build_feature_frame(history)
    manifest = portable_manifest()
    supported = manifest["supported"]
    abstained = manifest.get("abstained", {})

    results: dict[str, dict[str, Any]] = {}
    for outcome in OUTCOMES:
        horizons: dict[str, Any] = {}
        for horizon in HORIZON_DAYS:
            key = f"{outcome}:{horizon}"
            entry = supported.get(key)
            if entry is None:
                reason = abstained.get(key, {})
                horizons[horizon] = {
                    "status": "abstained",
                    "selected_model": None,
                    "models": {},
                    "reason": reason.get(
                        "reason",
                        "No model passed the event gate and usability floor for this horizon.",
                    ),
                    "detail": reason,
                }
                continue
            model_name = entry["selected_model"]
            spec = manifest["models"][key]
            if model_name == "discrete_time_hazard":
                base = design_matrix(features, spec["feature_columns"])
                raw = hazard_cumulative_incidence(spec, base, HORIZON_DAYS[horizon])
            elif model_name == "gradient_boosted_trees":
                base = design_matrix(features, spec["feature_columns"])
                raw = float(tree_probability(spec, base)[0])
            elif model_name == "temporal_gru":
                sequence, _, length = build_sequence(
                    matrix, spec["normaliser"], int(spec["max_visits"])
                )
                probabilities = gru_probabilities(spec, sequence, length)
                raw = float(probabilities[int(spec["horizon_index"])])
            else:  # pragma: no cover - guarded by the export script
                horizons[horizon] = {
                    "status": "abstained",
                    "selected_model": None,
                    "models": {},
                    "reason": (
                        f"Selected model '{model_name}' has no portable implementation, so "
                        "this horizon abstains rather than substituting another model."
                    ),
                }
                continue
            calibrated = apply_calibrator(entry["calibrator"], raw)
            horizons[horizon] = {
                "status": "simulated_estimate",
                "selected_model": model_name,
                "models": {
                    model_name: {
                        "raw_cumulative_incidence": round(raw, 6),
                        "calibrated_cumulative_incidence": round(calibrated, 6),
                        "calibration_method": entry["calibrator"]["method"],
                        "flags": _horizon_metrics(entry),
                    }
                },
                "selection_rule": entry.get("selection_rule"),
                "flags": _horizon_metrics(entry),
            }
        results[outcome] = {"horizons": horizons}

    return {
        "simulation_only": True,
        "clinical_use": "prohibited",
        "banner": manifest.get("banner", SIMULATION_BANNER_FALLBACK),
        "inference_backend": "numpy_portable",
        "history": {
            "visits": int(len(history)),
            "distinct_visit_times": int(history["days_before_index"].nunique()),
            "history_days": float(features["history_days"].iloc[0]),
            "visit_density_per_year": float(features["visit_density_per_year"].iloc[0]),
            "missingness_burden": round(float(features["missingness_burden"].iloc[0]), 4),
        },
        "outcomes": results,
        "disabled_outcomes": DISABLED_OUTCOMES,
        "interpretation": (
            "Simulated cumulative incidence for a synthetic history under a model fitted to "
            "synthetic data. It is not a patient's risk, it is not calibrated to any real "
            "population, and it must not inform care."
        ),
        "persistence": "none: the submitted history was scored in memory and discarded.",
    }
