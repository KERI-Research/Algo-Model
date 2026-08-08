"""Export the selected future-risk models to a portable NumPy/JSON artifact.

Runs **offline only**, on a machine that has the authoritative artifact plus
scikit-learn, PyTorch and joblib (the KERI repository checkout). It is never
imported by the server at request time.

Output, written into ``assets/future_risk/``:

* ``portable_models.json`` - manifest: which model is selected per
  outcome/horizon, its metrics and flags, the calibrator description, feature
  column order, abstention reasons, and the source-artifact provenance;
* ``portable_models.npz`` - every numeric parameter (logistic coefficients,
  imputer/scaler constants, exported tree nodes, GRU weights, isotonic
  thresholds);
* ``artifact_summary.json`` - the evaluation caveats and competing-outcome notes
  the dashboard displays;
* ``parity_report.json`` - measured agreement between the authoritative
  joblib/PyTorch scoring path and the portable NumPy path on representative
  synthetic histories.

Only the **selected** model per horizon is exported. A horizon whose selected
model cannot be ported is recorded as an abstention; no other model is
substituted and no probability is invented.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PORTABLE_SCHEMA_VERSION = "metaboguard-future-risk-portable-v1"

#: Selected models this exporter knows how to port exactly.
PORTABLE_MODEL_KINDS = ("discrete_time_hazard", "gradient_boosted_trees", "temporal_gru")

#: Horizon order the temporal model's head emits.
GRU_HORIZON_ORDER = ("1y", "3y", "5y")

#: Wide-interval flag threshold for the reported AUROC bootstrap interval.
WIDE_CI_WIDTH = 0.15


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


# ---------------------------------------------------------------------------
# Estimator export
# ---------------------------------------------------------------------------


def export_logistic_pipeline(pipeline: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """SimpleImputer(median, add_indicator) -> StandardScaler -> LogisticRegression."""
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    indicator = getattr(imputer, "indicator_", None)
    indicator_columns = (
        np.asarray(indicator.features_, dtype=np.int64) if indicator is not None else np.zeros(0, dtype=np.int64)
    )
    arrays[f"{prefix}/imputer_statistics"] = np.asarray(imputer.statistics_, dtype=np.float64)
    arrays[f"{prefix}/indicator_columns"] = indicator_columns
    arrays[f"{prefix}/scaler_mean"] = np.asarray(scaler.mean_, dtype=np.float64)
    arrays[f"{prefix}/scaler_scale"] = np.asarray(scaler.scale_, dtype=np.float64)
    arrays[f"{prefix}/coef"] = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    arrays[f"{prefix}/intercept"] = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    return {
        "kind": "logistic_pipeline",
        "array_prefix": prefix,
        "add_indicator": bool(getattr(imputer, "add_indicator", False)),
        "n_features_in": int(imputer.statistics_.shape[0]),
    }


def export_tree_pipeline(pipeline: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Flatten a HistGradientBoostingClassifier into node arrays."""
    model = pipeline.named_steps["model"]
    left: list[int] = []
    right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    missing_left: list[int] = []
    is_leaf: list[int] = []
    value: list[float] = []
    tree_start: list[int] = []

    for iteration in model._predictors:  # noqa: SLF001 - documented internal layout
        for predictor in iteration:
            nodes = predictor.nodes
            offset = len(left)
            tree_start.append(offset)
            for node in nodes:
                is_leaf.append(int(node["is_leaf"]))
                value.append(float(node["value"]))
                feature.append(int(node["feature_idx"]))
                threshold.append(float(node["num_threshold"]))
                missing_left.append(int(node["missing_go_to_left"]))
                left.append(int(node["left"]) + offset)
                right.append(int(node["right"]) + offset)

    arrays[f"{prefix}/left"] = np.asarray(left, dtype=np.int32)
    arrays[f"{prefix}/right"] = np.asarray(right, dtype=np.int32)
    arrays[f"{prefix}/feature"] = np.asarray(feature, dtype=np.int32)
    arrays[f"{prefix}/threshold"] = np.asarray(threshold, dtype=np.float64)
    arrays[f"{prefix}/missing_go_to_left"] = np.asarray(missing_left, dtype=np.int8)
    arrays[f"{prefix}/is_leaf"] = np.asarray(is_leaf, dtype=np.int8)
    arrays[f"{prefix}/value"] = np.asarray(value, dtype=np.float64)
    arrays[f"{prefix}/tree_start"] = np.asarray(tree_start, dtype=np.int32)
    return {
        "kind": "hist_gradient_boosting",
        "array_prefix": prefix,
        "baseline_prediction": float(np.asarray(model._baseline_prediction).reshape(-1)[0]),  # noqa: SLF001
        "trees": len(tree_start),
        "nodes": len(left),
    }


def export_gru(state_dict: dict[str, Any], prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Export the temporal model's weights (single-layer GRU plus MLP head)."""
    def take(name: str) -> np.ndarray:
        return np.asarray(state_dict[name], dtype=np.float64)

    arrays[f"{prefix}/input_projection_weight"] = take("input_projection.weight")
    arrays[f"{prefix}/input_projection_bias"] = take("input_projection.bias")
    arrays[f"{prefix}/gru_weight_ih"] = take("gru.weight_ih_l0")
    arrays[f"{prefix}/gru_weight_hh"] = take("gru.weight_hh_l0")
    arrays[f"{prefix}/gru_bias_ih"] = take("gru.bias_ih_l0")
    arrays[f"{prefix}/gru_bias_hh"] = take("gru.bias_hh_l0")
    arrays[f"{prefix}/head0_weight"] = take("head.0.weight")
    arrays[f"{prefix}/head0_bias"] = take("head.0.bias")
    arrays[f"{prefix}/head2_weight"] = take("head.2.weight")
    arrays[f"{prefix}/head2_bias"] = take("head.2.bias")
    return {"kind": "temporal_gru", "array_prefix": prefix}


def export_calibrator(calibrator: dict[str, Any], prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if calibrator["method"] == "isotonic":
        model = calibrator["model"]
        arrays[f"{prefix}/x_thresholds"] = np.asarray(model.X_thresholds_, dtype=np.float64)
        arrays[f"{prefix}/y_thresholds"] = np.asarray(model.y_thresholds_, dtype=np.float64)
        return {
            "method": "isotonic",
            "array_prefix": prefix,
            "n": int(calibrator.get("n", 0)),
            "points": int(np.asarray(model.X_thresholds_).size),
        }
    model = calibrator["model"]
    return {
        "method": "platt_logistic",
        "coef": float(np.asarray(model.coef_).reshape(-1)[0]),
        "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
        "n": int(calibrator.get("n", 0)),
    }


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def _flags(metrics: dict[str, Any]) -> dict[str, Any]:
    interval = (
        metrics.get("auroc_95ci")
        or metrics.get("auroc_ci")
        or metrics.get("auroc_confidence_interval")
    )
    width = None
    if isinstance(interval, (list, tuple)) and len(interval) == 2 and all(
        isinstance(bound, (int, float)) for bound in interval
    ):
        width = float(interval[1]) - float(interval[0])
    payload = dict(metrics)
    payload["auroc_95ci"] = list(interval) if isinstance(interval, (list, tuple)) else None
    payload["auroc_ci_width"] = round(width, 4) if width is not None else None
    payload["wide_confidence_interval"] = bool(width is not None and width >= WIDE_CI_WIDTH)
    return payload


def build_manifest(bundle: dict[str, Any], metadata: dict[str, Any], results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    feature_columns = list(metadata["feature_columns"])
    selection = metadata["selection"]
    supported: dict[str, Any] = {}
    abstained: dict[str, Any] = {}
    models: dict[str, Any] = {}

    for outcome in metadata["outcomes_trained"]:
        outcome_bundle = bundle[outcome]
        for horizon in GRU_HORIZON_ORDER:
            key = f"{outcome}:{horizon}"
            entry = selection.get(key)
            if not entry or not entry.get("selected_model"):
                abstained[key] = {
                    "reason": (
                        "No model passed the event gate and usability floor for this horizon, "
                        "so the deployment abstains."
                    ),
                    "gate": _json_safe((entry or {}).get("metrics", {})),
                }
                continue
            model_name = entry["selected_model"]
            if model_name not in PORTABLE_MODEL_KINDS:
                abstained[key] = {
                    "reason": (
                        f"Selected model '{model_name}' has no portable implementation; the "
                        "deployment abstains rather than substituting another model."
                    )
                }
                continue
            prefix = f"{outcome}/{horizon}/{model_name}"
            calibrator_key = f"{model_name}:{horizon}"
            calibrator = outcome_bundle["calibrators"].get(calibrator_key)
            if calibrator is None:
                abstained[key] = {
                    "reason": "No calibrator was fitted for the selected model; abstaining.",
                }
                continue

            if model_name == "discrete_time_hazard":
                hazard = outcome_bundle["hazard"]
                if hazard.get("status") != "fitted":
                    abstained[key] = {"reason": "Hazard model was not fitted; abstaining."}
                    continue
                event_spec = export_logistic_pipeline(hazard["model"], f"{prefix}/event", arrays)
                competing_spec = (
                    export_logistic_pipeline(hazard["competing_model"], f"{prefix}/competing", arrays)
                    if hazard.get("competing_model") is not None
                    else None
                )
                models[key] = {
                    "kind": "discrete_time_hazard",
                    "feature_columns": list(hazard["design_columns"][:-1]),
                    "interval_days": int(hazard["interval_days"]),
                    "event": event_spec,
                    "competing": competing_spec,
                    "competing_handled": bool(hazard.get("competing_handled")),
                    "train_person_intervals": int(hazard.get("train_person_intervals", 0)),
                    "train_events": int(hazard.get("train_events", 0)),
                }
            elif model_name == "gradient_boosted_trees":
                baseline = outcome_bundle["baselines"].get(horizon, {})
                pipeline = (baseline.get("models") or {}).get("gradient_boosted_trees")
                if pipeline is None:
                    abstained[key] = {"reason": "Tree baseline missing; abstaining."}
                    continue
                spec = export_tree_pipeline(pipeline, prefix, arrays)
                spec["feature_columns"] = feature_columns
                models[key] = spec
            else:  # temporal_gru
                state = outcome_bundle.get("temporal_state_dict")
                meta = outcome_bundle.get("temporal_meta") or {}
                normaliser = outcome_bundle.get("temporal_normaliser") or {}
                if not state:
                    abstained[key] = {"reason": "Temporal state dict missing; abstaining."}
                    continue
                spec = export_gru(state, prefix, arrays)
                spec["normaliser"] = {
                    name: [float(centre), float(spread)]
                    for name, (centre, spread) in normaliser.items()
                }
                spec["max_visits"] = int(meta.get("max_visits", 8))
                spec["input_dim"] = int(meta.get("input_dim", 23))
                spec["hidden_dim"] = int(meta.get("hidden_dim", 48))
                spec["horizon_index"] = GRU_HORIZON_ORDER.index(horizon)
                spec["architecture"] = outcome_bundle.get("temporal_architecture")
                models[key] = spec

            supported[key] = {
                "selected_model": model_name,
                "selection_rule": entry.get("rule"),
                "metrics": _flags(_json_safe(entry.get("metrics", {}))),
                "calibrator": export_calibrator(calibrator, f"{prefix}/calibrator", arrays),
                "calibration_method": calibrator["method"],
            }

    manifest = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "banner": metadata.get("banner"),
        "simulation_only": True,
        "clinical_use": "prohibited",
        "feature_columns": feature_columns,
        "portable_model_kinds": list(PORTABLE_MODEL_KINDS),
        "source_artifact": {
            "artifact_version": metadata.get("artifact_version"),
            "code_version": metadata.get("code_version"),
            "created_at": metadata.get("created_at"),
            "capability_state": metadata.get("capability_state"),
            "schema_version": metadata.get("schema_version"),
            "package_versions": metadata.get("package_versions"),
            "results_fingerprint": metadata.get("results_fingerprint"),
        },
        "supported": _json_safe(supported),
        "abstained": _json_safe(abstained),
        "models": _json_safe(models),
        "note": (
            "Only the selected model per horizon is exported. Unsupported horizons abstain; "
            "no model is ever substituted for another and no probability is fabricated."
        ),
    }
    del results
    return manifest, arrays


def build_summary(metadata: dict[str, Any], results_summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluation caveats the dashboard shows next to every simulated estimate."""
    return {
        "banner": metadata.get("banner"),
        "capability_state": metadata.get("capability_state"),
        "data_source": "synthetic longitudinal cohort (Synthea-derived, pooled)",
        "competing_outcomes": (
            "Death from another cause competes with the outcome of interest. The discrete-time "
            "hazard model is cause-specific and subtracts the competing hazard at each "
            "interval, so a patient who dies first cannot be counted as a future case. Models "
            "that ignore competing risk overstate incidence."
        ),
        "underpowered_test_split": (
            "Every reported horizon has fewer than 25 events in the held-out split. Metrics "
            "computed on so few events are unstable, and the ranking between models is not "
            "meaningfully resolved."
        ),
        "wide_confidence_intervals": (
            "Bootstrap AUROC intervals span a wide range, so the point estimate should not be "
            "read as the model's discrimination. An interval that includes 0.5 is compatible "
            "with no discrimination at all."
        ),
        "temporal_admissibility": _json_safe(metadata.get("temporal_admissibility", {})),
        "caveats": [
            "Synthetic data only: nothing here estimates real patient risk.",
            "Not calibrated to any real population; calibration is in-simulation only.",
            "Test splits are underpowered; treat all metrics as software verification.",
            "AUROC values are simulation diagnostics, not clinical performance.",
            "Competing death is modelled cause-specifically, but the simulated event rates are "
            "not epidemiologically valid.",
            "No external validation, no prospective specimens, no clinical utility analysis.",
        ],
        "results_summary_keys": sorted(results_summary)[:40] if isinstance(results_summary, dict) else [],
    }


# ---------------------------------------------------------------------------
# Parity harness
# ---------------------------------------------------------------------------


def representative_histories(count: int = 40, seed: int = 20260805) -> list[pd.DataFrame]:
    """Deterministic synthetic histories spanning visit counts and metabolic severity."""
    rng = np.random.default_rng(seed)
    from .future_risk import HISTORY_FEATURES

    anchors = {
        "DEMO_RIDAGEYR": (34.0, 78.0),
        "BMX_BMXBMI": (20.0, 43.0),
        "BMX_BMXWAIST": (72.0, 128.0),
        "BMX_BMXWT": (56.0, 122.0),
        "GHB_LBXGH": (4.8, 9.4),
        "GLU_LBXGLU": (78.0, 190.0),
        "INS_LBXIN": (3.0, 34.0),
        "TCHOL_LBXTC": (140.0, 268.0),
        "HDL_LBDHDD": (28.0, 74.0),
        "TRIGLY_LBXTR": (55.0, 340.0),
        "BPX_SYSTOLIC": (102.0, 168.0),
    }
    histories: list[pd.DataFrame] = []
    for index in range(count):
        visits = int(rng.integers(2, 10))
        gaps = rng.integers(120, 620, size=visits - 1)
        offsets = [0]
        for gap in gaps[::-1]:
            offsets.insert(0, offsets[0] + int(gap))
        rows = []
        base = {name: float(rng.uniform(low, high)) for name, (low, high) in anchors.items()}
        drift = float(rng.uniform(-0.6, 1.4))
        for position, days in enumerate(offsets):
            years = days / 365.25
            row: dict[str, Any] = {"visit_index": position, "days_before_index": float(days)}
            for name in HISTORY_FEATURES:
                # Deterministic missingness, including an all-missing feature sometimes.
                if name != "DEMO_RIDAGEYR" and rng.random() < (0.3 if index % 5 else 0.6):
                    row[name] = np.nan
                    continue
                if name == "DEMO_RIDAGEYR":
                    row[name] = round(base[name] - years, 2)
                else:
                    row[name] = round(base[name] * (1.0 - 0.01 * drift * years), 3)
            rows.append(row)
        histories.append(pd.DataFrame(rows))
    return histories


def authoritative_scores(
    bundle: dict[str, Any], metadata: dict[str, Any], histories: list[pd.DataFrame]
) -> list[dict[str, float]]:
    """Score the histories with the joblib pipelines and the PyTorch model."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
    import future_risk_models as frm  # noqa: E402  (needs sklearn + torch)
    import torch  # noqa: E402

    from .future_risk import HORIZON_DAYS, build_feature_frame

    feature_columns = list(metadata["feature_columns"])
    scored: list[dict[str, float]] = []
    for history in histories:
        matrix, features = build_feature_frame(history)
        row: dict[str, float] = {}
        for outcome in metadata["outcomes_trained"]:
            outcome_bundle = bundle[outcome]
            for horizon, days in HORIZON_DAYS.items():
                entry = metadata["selection"].get(f"{outcome}:{horizon}")
                if not entry or not entry.get("selected_model"):
                    continue
                model_name = entry["selected_model"]
                if model_name == "discrete_time_hazard":
                    value = float(
                        frm.hazard_cumulative_incidence(outcome_bundle["hazard"], features, days)[0]
                    )
                elif model_name == "gradient_boosted_trees":
                    pipeline = outcome_bundle["baselines"][horizon]["models"][model_name]
                    value = float(
                        pipeline.predict_proba(
                            features.reindex(columns=feature_columns).astype(float).to_numpy()
                        )[0, 1]
                    )
                elif model_name == "temporal_gru":
                    meta = outcome_bundle["temporal_meta"]
                    model = frm.reload_temporal_model(
                        outcome_bundle["temporal_state_dict"],
                        int(meta["input_dim"]),
                        int(meta["hidden_dim"]),
                        len(GRU_HORIZON_ORDER),
                    )
                    normaliser = {
                        name: tuple(value) for name, value in outcome_bundle["temporal_normaliser"].items()
                    }
                    visit_frame = matrix.copy()
                    visit_frame["patient_id"] = "simulated-history"
                    sequences, padding, _, _, _ = frm.build_sequences(
                        visit_frame,
                        ["simulated-history"],
                        frm.FutureRiskConfig(),
                        normaliser=normaliser,
                    )
                    with torch.no_grad():
                        logits = model(
                            torch.tensor(sequences, dtype=torch.float32),
                            torch.tensor(padding, dtype=torch.float32),
                        )
                        probabilities = torch.sigmoid(logits).numpy()[0]
                    value = float(probabilities[GRU_HORIZON_ORDER.index(horizon)])
                else:
                    continue
                calibrator = outcome_bundle["calibrators"][f"{model_name}:{horizon}"]
                calibrated = float(
                    frm.apply_calibrator(calibrator, np.array([value], dtype=float))[0]
                )
                row[f"{outcome}:{horizon}:raw"] = value
                row[f"{outcome}:{horizon}:calibrated"] = calibrated
        scored.append(row)
    return scored


def portable_scores(histories: list[pd.DataFrame]) -> list[dict[str, float]]:
    from .future_risk import (
        HORIZON_DAYS,
        apply_calibrator,
        build_feature_frame,
        build_sequence,
        design_matrix,
        gru_probabilities,
        hazard_cumulative_incidence,
        portable_manifest,
        tree_probability,
    )

    manifest = portable_manifest()
    scored: list[dict[str, float]] = []
    for history in histories:
        matrix, features = build_feature_frame(history)
        row: dict[str, float] = {}
        for key, entry in manifest["supported"].items():
            outcome, horizon = key.split(":")
            spec = manifest["models"][key]
            model_name = entry["selected_model"]
            if model_name == "discrete_time_hazard":
                raw = hazard_cumulative_incidence(
                    spec, design_matrix(features, spec["feature_columns"]), HORIZON_DAYS[horizon]
                )
            elif model_name == "gradient_boosted_trees":
                raw = float(
                    tree_probability(spec, design_matrix(features, spec["feature_columns"]))[0]
                )
            else:
                sequence, _, length = build_sequence(
                    matrix, spec["normaliser"], int(spec["max_visits"])
                )
                raw = float(gru_probabilities(spec, sequence, length)[int(spec["horizon_index"])])
            row[f"{key}:raw"] = raw
            row[f"{key}:calibrated"] = apply_calibrator(entry["calibrator"], raw)
        scored.append(row)
    return scored


def compare(authoritative: list[dict[str, float]], portable: list[dict[str, float]]) -> dict[str, Any]:
    per_key: dict[str, dict[str, float]] = {}
    worst = 0.0
    for reference, candidate in zip(authoritative, portable):
        for key, value in reference.items():
            other = candidate.get(key)
            if other is None:
                per_key.setdefault(key, {"max_abs_difference": float("inf"), "compared": 0})
                per_key[key]["missing_in_portable"] = True
                continue
            difference = abs(float(value) - float(other))
            bucket = per_key.setdefault(key, {"max_abs_difference": 0.0, "compared": 0})
            bucket["max_abs_difference"] = max(bucket["max_abs_difference"], difference)
            bucket["compared"] += 1
            worst = max(worst, difference)
    tolerance = 1e-6
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "histories_compared": len(authoritative),
        "tolerance": tolerance,
        "max_abs_difference": worst,
        "per_output": {
            key: {
                "max_abs_difference": round(value["max_abs_difference"], 12),
                "compared": value["compared"],
                "within_tolerance": value["max_abs_difference"] <= tolerance,
            }
            for key, value in sorted(per_key.items())
        },
        "verdict": "parity" if worst <= tolerance else "MISMATCH",
        "method": (
            "Deterministic synthetic histories (varying visit counts, gaps, severity and "
            "missingness) scored with the authoritative joblib pipelines and PyTorch model, "
            "then with the portable NumPy path. Raw and calibrated values compared."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def export(artifact_dir: Path, output_dir: Path, histories: int = 40) -> dict[str, Any]:
    import joblib

    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    results_summary_path = artifact_dir / "results_summary.json"
    results_summary = (
        json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    )
    bundle = joblib.load(artifact_dir / "future_risk_models.joblib")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arrays = build_manifest(bundle, metadata, results_summary)
    (output_dir / "portable_models.json").write_text(json.dumps(manifest, indent=1))
    np.savez_compressed(output_dir / "portable_models.npz", **arrays)
    (output_dir / "artifact_summary.json").write_text(
        json.dumps(build_summary(metadata, results_summary), indent=1)
    )

    # Parity must be measured after the portable artifact is on disk.
    from .future_risk import portable_arrays, portable_manifest

    portable_manifest.cache_clear()
    portable_arrays.cache_clear()
    samples = representative_histories(histories)
    report = compare(
        authoritative_scores(bundle, metadata, samples), portable_scores(samples)
    )
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=1))
    return report
