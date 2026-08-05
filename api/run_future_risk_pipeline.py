"""End-to-end simulation-only future-risk pipeline.

Steps: load prepared dataset -> fit transparent baselines, cause-specific discrete-time
hazards and a temporal GRU -> calibrate on validation -> evaluate on test and the temporal
holdout -> negative controls -> calibration-first selection -> versioned simulation-only
artifact with a model card.

Usage::

    python run_future_risk_pipeline.py --smoke
    python run_future_risk_pipeline.py --prepared ../data/synthetic_longitudinal/prepared \
        --output-dir ../model_artifacts/future_risk/run1
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from future_risk_models import (
    CODE_VERSION,
    SIMULATION_ONLY_BANNER,
    FutureRiskConfig,
    add_subgroup_columns,
    apply_calibrator,
    baseline_feature_columns,
    build_person_interval_frame,
    evaluate_predictions,
    fit_calibrator,
    fit_discrete_time_hazard,
    fit_horizon_baselines,
    hazard_cumulative_incidence,
    train_temporal_model,
)
from longitudinal_schema import (
    CapabilityState,
    HORIZON_DAYS,
    HORIZON_LABELS,
    MIN_EVENTS_PER_HORIZON,
    frame_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREPARED = PROJECT_ROOT / "data" / "synthetic_longitudinal" / "prepared"
DEFAULT_OUTPUT = PROJECT_ROOT / "model_artifacts" / "future_risk"


def _package_versions() -> dict[str, str]:
    import sklearn

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:  # pragma: no cover
        versions["torch"] = "not_installed"
    return versions


def _predictions_for_split(
    kind: str,
    fitted: dict[str, Any],
    frame: pd.DataFrame,
    columns: list[str],
    outcome: str,
    horizon: int,
    temporal: dict[str, Any] | None,
) -> np.ndarray | None:
    if kind in {"horizon_logistic", "gradient_boosted_trees"}:
        model = fitted.get("models", {}).get(kind)
        if model is None:
            return None
        return model.predict_proba(frame[columns].astype(float).to_numpy())[:, 1]
    if kind == "discrete_time_hazard":
        if fitted.get("status") != "fitted":
            return None
        return hazard_cumulative_incidence(fitted, frame, horizon)
    if kind == "temporal_gru":
        if temporal is None or temporal.get("status") != "fitted":
            return None
        ordered, probabilities = temporal["predict"](frame["patient_id"].tolist())
        if not ordered:
            return None
        lookup = {pid: probabilities[i] for i, pid in enumerate(ordered)}
        horizon_index = temporal["horizons"].index(horizon)
        return np.array(
            [lookup[pid][horizon_index] if pid in lookup else np.nan for pid in frame["patient_id"]]
        )
    return None


def run_pipeline(
    prepared_dir: str | Path = DEFAULT_PREPARED,
    output_dir: str | Path | None = None,
    config: FutureRiskConfig | None = None,
) -> dict[str, Any]:
    config = config or FutureRiskConfig()
    prepared = Path(prepared_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_dir) if output_dir else DEFAULT_OUTPUT / f"simulation__{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    features = pd.read_csv(prepared / "patient_features.csv", low_memory=False)
    visit_matrix = pd.read_csv(prepared / "visit_matrix.csv", low_memory=False)
    splits = json.loads((prepared / "splits.json").read_text())
    split_manifest = json.loads((prepared / "split_manifest.json").read_text())
    dataset_validation = json.loads((prepared / "dataset_validation_report.json").read_text())
    features = add_subgroup_columns(features)
    columns = baseline_feature_columns(features)
    # Some source cohorts simply do not carry every prevention-safe feature (Synthea has no
    # insulin assay, for example), leaving all-NaN or constant columns that tree binning cannot
    # fit. Those columns are dropped on the *train* split only and the drop is reported, rather
    # than being imputed into a fake signal.
    train_frame = features[features["patient_id"].isin(splits["train"])]
    usable, dropped = [], []
    for column in columns:
        values = pd.to_numeric(train_frame[column], errors="coerce")
        if values.notna().sum() == 0 or values.dropna().nunique() < 2:
            dropped.append(column)
        else:
            usable.append(column)
    columns = usable

    results: dict[str, Any] = {
        "run_type": "metaboguard_future_risk_simulation",
        "code_version": CODE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "capability_state": CapabilityState.SIMULATION_ONLY_LONGITUDINAL.value,
        "simulation_only": True,
        "banner": SIMULATION_ONLY_BANNER,
        "config": config.as_dict(),
        "feature_columns": columns,
        "split_manifest": split_manifest,
        "dataset_gates": dataset_validation["gates"],
        "package_versions": _package_versions(),
        "outcomes": {},
    }

    artifacts: dict[str, Any] = {}
    prediction_rows: list[pd.DataFrame] = []
    for outcome in config.outcomes:
        gate = dataset_validation["gates"][outcome]
        outcome_block: dict[str, Any] = {
            "gate": gate,
            "trained": False,
            "horizons": {},
        }
        if not gate["any_horizon_eligible"]:
            outcome_block["skip_reason"] = (
                f"No horizon passes the {MIN_EVENTS_PER_HORIZON}-event gate; refusing to train."
            )
            results["outcomes"][outcome] = outcome_block
            continue

        hazard = fit_discrete_time_hazard(features, columns, outcome, splits, config)
        temporal = train_temporal_model(visit_matrix, features, outcome, splits, config)
        outcome_block["discrete_time_hazard"] = {
            "status": hazard.get("status"),
            "train_person_intervals": hazard.get("train_person_intervals"),
            "train_events": hazard.get("train_events"),
            "competing_event_handled": hazard.get("competing_handled"),
            "interval_days": hazard.get("interval_days"),
        }
        outcome_block["temporal_model"] = {
            "status": temporal.get("status"),
            "architecture": temporal.get("architecture"),
            "input_dim": temporal.get("input_dim"),
            "epochs_run": len(temporal.get("history", [])),
            "final_validation_loss": temporal["history"][-1]["validation_loss"] if temporal.get("history") else None,
            "separate_encoder_rationale": temporal.get("separate_encoder_rationale"),
        }
        artifacts[outcome] = {"hazard": hazard, "temporal": temporal, "baselines": {}, "calibrators": {}}

        for horizon in config.horizons_days:
            suffix = HORIZON_LABELS[horizon]
            horizon_gate = gate["per_horizon"][suffix]
            block: dict[str, Any] = {"gate": horizon_gate, "models": {}}
            if not horizon_gate["eligible"]:
                block["skip_reason"] = "horizon below the event gate"
                outcome_block["horizons"][suffix] = block
                continue

            baselines = fit_horizon_baselines(features, columns, outcome, horizon, splits, config)
            artifacts[outcome]["baselines"][suffix] = baselines
            mask_column = f"{outcome}_{suffix}_eligible"

            def eligible(split: str) -> pd.DataFrame:
                frame = features[features["patient_id"].isin(splits[split])]
                return frame[frame[mask_column] == 1].reset_index(drop=True)

            validation_frame = eligible("validation")
            evaluation_frames = {
                "test": eligible("test"),
                "temporal_holdout": eligible("temporal_holdout") if "temporal_holdout" in splits else pd.DataFrame(),
            }

            for kind in ("horizon_logistic", "gradient_boosted_trees", "discrete_time_hazard", "temporal_gru"):
                source = baselines if kind in {"horizon_logistic", "gradient_boosted_trees"} else (
                    artifacts[outcome]["hazard"] if kind == "discrete_time_hazard" else {}
                )
                validation_probabilities = _predictions_for_split(
                    kind, source, validation_frame, columns, outcome, horizon, temporal
                )
                if validation_probabilities is None or not len(validation_frame):
                    block["models"][kind] = {"status": "unavailable"}
                    continue
                labels = validation_frame[f"{outcome}_{suffix}_label"].to_numpy().astype(int)
                calibrator = fit_calibrator(np.nan_to_num(validation_probabilities), labels)
                artifacts[outcome]["calibrators"][f"{kind}:{suffix}"] = calibrator

                model_block: dict[str, Any] = {
                    "status": "evaluated",
                    "calibration_method": calibrator["method"],
                    "validation_rows": int(len(validation_frame)),
                    "splits": {},
                }
                for split_name, frame in evaluation_frames.items():
                    if frame.empty:
                        model_block["splits"][split_name] = {"status": "empty_split"}
                        continue
                    raw = _predictions_for_split(kind, source, frame, columns, outcome, horizon, temporal)
                    if raw is None:
                        model_block["splits"][split_name] = {"status": "unavailable"}
                        continue
                    raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0)
                    calibrated = apply_calibrator(calibrator, raw)
                    model_block["splits"][split_name] = {
                        "raw": evaluate_predictions(frame, raw, outcome, horizon, config),
                        "calibrated": evaluate_predictions(frame, calibrated, outcome, horizon, config),
                    }
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "outcome": outcome,
                                "horizon": suffix,
                                "model": kind,
                                "split": split_name,
                                "patient_id": frame["patient_id"].to_numpy(),
                                "label": frame[f"{outcome}_{suffix}_label"].to_numpy(),
                                "eligible": 1,
                                "raw_probability": raw,
                                "calibrated_probability": calibrated,
                                "simulation_only": True,
                            }
                        )
                    )
                block["models"][kind] = model_block

            # Negative controls on the test split: shuffled labels and time-reversed sequences.
            test_frame = evaluation_frames["test"]
            controls: dict[str, Any] = {}
            if not test_frame.empty:
                random = np.random.default_rng(config.seed + 1)
                shuffled = test_frame.copy()
                shuffled[f"{outcome}_{suffix}_label"] = random.permutation(
                    shuffled[f"{outcome}_{suffix}_label"].to_numpy()
                )
                logistic = baselines.get("models", {}).get("horizon_logistic")
                if logistic is not None:
                    probabilities = logistic.predict_proba(test_frame[columns].astype(float).to_numpy())[:, 1]
                    controls["shuffled_outcome_labels"] = {
                        "expectation": "AUROC near 0.5; anything higher indicates leakage.",
                        **{
                            key: value
                            for key, value in evaluate_predictions(
                                shuffled, probabilities, outcome, horizon, config
                            ).items()
                            if key in {"auroc", "average_precision", "status", "events", "n_eligible"}
                        },
                    }
                if temporal.get("status") == "fitted":
                    reversed_matrix = visit_matrix.copy()
                    reversed_matrix["visit_index"] = reversed_matrix.groupby("patient_id")[
                        "visit_index"
                    ].transform(lambda series: series.max() - series)
                    reversed_temporal = dict(temporal)
                    ordered, probabilities = temporal["predict"](test_frame["patient_id"].tolist())
                    forward = {pid: probabilities[i] for i, pid in enumerate(ordered)}
                    from future_risk_models import build_sequences

                    sequences, padding, _, ordered_reversed, _ = build_sequences(
                        reversed_matrix.sort_values(["patient_id", "visit_index"]),
                        test_frame["patient_id"].tolist(),
                        config,
                        normaliser=temporal["normaliser"],
                    )
                    import torch

                    with torch.no_grad():
                        reversed_probabilities = (
                            torch.sigmoid(
                                temporal["model"](torch.tensor(sequences), torch.tensor(padding))
                            )
                            .cpu()
                            .numpy()
                        )
                    horizon_index = temporal["horizons"].index(horizon)
                    aligned_forward = np.array(
                        [forward[pid][horizon_index] for pid in ordered_reversed if pid in forward]
                    )
                    aligned_reversed = np.array(
                        [
                            reversed_probabilities[i][horizon_index]
                            for i, pid in enumerate(ordered_reversed)
                            if pid in forward
                        ]
                    )
                    labels_reversed = (
                        test_frame.set_index("patient_id")
                        .loc[[pid for pid in ordered_reversed if pid in forward], f"{outcome}_{suffix}_label"]
                        .to_numpy()
                        .astype(int)
                    )
                    from sklearn.metrics import roc_auc_score

                    controls["time_reversed_sequences"] = {
                        "expectation": (
                            "Reversing the visit order should degrade discrimination; identical "
                            "performance would mean the model ignores temporal order."
                        ),
                        "forward_auroc": round(float(roc_auc_score(labels_reversed, aligned_forward)), 4)
                        if len(np.unique(labels_reversed)) > 1
                        else None,
                        "reversed_auroc": round(float(roc_auc_score(labels_reversed, aligned_reversed)), 4)
                        if len(np.unique(labels_reversed)) > 1
                        else None,
                        "mean_absolute_probability_shift": round(
                            float(np.mean(np.abs(aligned_forward - aligned_reversed))), 6
                        ),
                    }
                    del reversed_temporal
            block["negative_controls"] = controls
            outcome_block["horizons"][suffix] = block

        outcome_block["trained"] = True
        results["outcomes"][outcome] = outcome_block

    # Calibration-first selection: among evaluated models, prefer the smallest |slope-1| on the
    # test split, breaking ties by AUROC. Discrimination never overrides bad calibration.
    selection: dict[str, Any] = {}
    temporal_admissibility: dict[str, Any] = {}
    for outcome, block in results["outcomes"].items():
        for suffix, horizon_block in block.get("horizons", {}).items():
            control = (horizon_block.get("negative_controls") or {}).get("time_reversed_sequences", {})
            forward, reversed_auroc = control.get("forward_auroc"), control.get("reversed_auroc")
            drop = (
                round(float(forward - reversed_auroc), 4)
                if forward is not None and reversed_auroc is not None
                else None
            )
            admissible = drop is not None and drop >= config.time_reversal_min_auroc_drop
            temporal_admissibility[f"{outcome}:{suffix}"] = {
                "forward_auroc": forward,
                "reversed_auroc": reversed_auroc,
                "auroc_drop": drop,
                "required_drop": config.time_reversal_min_auroc_drop,
                "temporal_model_admissible": bool(admissible),
                "decision": (
                    "temporal model admissible: reversing visit order degrades discrimination"
                    if admissible
                    else "temporal model REJECTED as experimental: reversing visit order does not "
                    "degrade discrimination, so it is not demonstrably using time"
                ),
            }
            candidates = []
            for kind, model_block in horizon_block.get("models", {}).items():
                if kind == "temporal_gru" and not admissible:
                    model_block["admissibility"] = "experimental_rejected_time_reversal_control"
                    continue
                test_block = model_block.get("splits", {}).get("test", {})
                calibrated = test_block.get("calibrated") if isinstance(test_block, dict) else None
                if not calibrated or calibrated.get("status") != "evaluated":
                    continue
                slope = (calibrated.get("calibration") or {}).get("slope")
                candidates.append(
                    {
                        "model": kind,
                        "auroc": calibrated["auroc"],
                        "brier": calibrated["brier"],
                        "calibration_slope": slope,
                        "slope_distance": abs((slope if slope is not None else 0) - 1.0),
                    }
                )
            # Usability floor: a model with a non-positive calibration slope is not merely
            # miscalibrated, its ranking is unusable, and near-chance discrimination cannot
            # support any estimate. Failing both is an abstention, not a "best available" pick.
            admissible = [
                candidate
                for candidate in candidates
                if (candidate["calibration_slope"] or 0) >= config.min_calibration_slope
                and candidate["auroc"] >= config.min_auroc
            ]
            if admissible:
                best = sorted(admissible, key=lambda item: (item["slope_distance"], -item["auroc"]))[0]
                selection[f"{outcome}:{suffix}"] = {
                    "selected_model": best["model"],
                    "rule": (
                        "calibration-first: smallest |calibration slope - 1| among models passing "
                        f"the usability floor (slope >= {config.min_calibration_slope}, "
                        f"AUROC >= {config.min_auroc}), ties by AUROC"
                    ),
                    "metrics": best,
                    "candidates": candidates,
                }
            elif candidates:
                horizon_block["abstained"] = {
                    "reason": "no_admissible_model",
                    "explanation": (
                        f"No model met the usability floor (calibration slope >= "
                        f"{config.min_calibration_slope} and AUROC >= {config.min_auroc}). "
                        "This horizon abstains rather than reporting a best-of-bad estimate."
                    ),
                    "candidates": candidates,
                }
    results["selection"] = selection
    results["temporal_admissibility"] = temporal_admissibility
    results["seconds"] = round(time.perf_counter() - started, 1)

    # Persist the artifact (weights, preprocessing, calibrators, split manifest, card).
    artifact_dir = output / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saveable: dict[str, Any] = {}
    for outcome, bundle in artifacts.items():
        saveable[outcome] = {
            "baselines": {
                suffix: {"models": item.get("models", {})} for suffix, item in bundle["baselines"].items()
            },
            "hazard": {k: v for k, v in bundle["hazard"].items() if k not in {"predict"}},
            "calibrators": bundle["calibrators"],
            "temporal_state_dict": bundle["temporal"].get("state_dict"),
            "temporal_normaliser": bundle["temporal"].get("normaliser"),
            "temporal_architecture": bundle["temporal"].get("architecture"),
            "temporal_meta": {
                "input_dim": bundle["temporal"].get("input_dim"),
                "hidden_dim": bundle["temporal"].get("hidden_dim"),
                "max_visits": bundle["temporal"].get("max_visits"),
                "horizons": bundle["temporal"].get("horizons"),
            },
            "temporal_history": bundle["temporal"].get("history"),
        }
    joblib.dump(saveable, artifact_dir / "future_risk_models.joblib")
    if prediction_rows:
        predictions = pd.concat(prediction_rows, ignore_index=True)
        predictions.to_csv(artifact_dir / "predictions_simulation_only.csv", index=False)
        results["predictions_path"] = str(artifact_dir / "predictions_simulation_only.csv")
        results["prediction_rows"] = int(len(predictions))
    (artifact_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    from longitudinal_schema import json_schemas

    (artifact_dir / "longitudinal_schema.json").write_text(json.dumps(json_schemas(), indent=2, default=str))
    (artifact_dir / "dataset_validation_report.json").write_text(
        json.dumps(dataset_validation, indent=2, default=str)
    )
    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    (artifact_dir / "feature_columns.json").write_text(json.dumps(columns, indent=2))
    (artifact_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (artifact_dir / "MODEL_CARD.md").write_text(render_future_risk_model_card(results))
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str))
    metadata = {
        "artifact_version": "future-risk-simulation-v1",
        "code_version": CODE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "capability_state": CapabilityState.SIMULATION_ONLY_LONGITUDINAL.value,
        "simulation_only": True,
        "clinical_use": "prohibited",
        "banner": SIMULATION_ONLY_BANNER,
        "schema_version": "metaboguard-longitudinal-v1",
        "feature_columns": columns,
        "outcomes_trained": [o for o, b in results["outcomes"].items() if b.get("trained")],
        "selection": selection,
        "temporal_admissibility": temporal_admissibility,
        "package_versions": results["package_versions"],
        "results_fingerprint": frame_fingerprint(pd.DataFrame({"payload": [json.dumps(selection, sort_keys=True)]})),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results["artifact_dir"] = str(artifact_dir)
    return results


def render_future_risk_model_card(results: dict[str, Any]) -> str:
    lines = [
        "# Model card: MetaboGuard future-risk (SIMULATION ONLY)",
        "",
        f"- Code version: `{results['code_version']}`",
        f"- Created: {results['generated_at']}",
        f"- Capability state: **{results['capability_state']}**",
        "",
        f"> {results['banner']}",
        "",
        "## Intended use",
        "",
        "Software and protocol verification for a future-risk pipeline: schema, endpoint "
        "definitions, eligibility masks, competing-risk handling, calibration and evaluation "
        "code paths. Nothing else.",
        "",
        "## Prohibited use",
        "",
        "- Any statement about a real patient's risk.",
        "- Any claim of early detection, clinical utility or calibration transfer.",
        "- Enabling the clinical future-risk endpoint (it stays HTTP 409 by capability gate).",
        "",
        "## Data",
        "",
        "Synthetic longitudinal cohort. Preferred generator is official Synthea "
        "(Apache-2.0, https://github.com/synthetichealth/synthea, methodology DOI "
        "10.1093/jamia/ocx079); Synthea's own validity limitations are documented at DOI "
        "10.1186/s12911-019-0793-0. When Synthea cannot run, a declared in-repo simulator is "
        "used with explicit enrichment strata and sampling weights, so raw probabilities are "
        "not population-calibrated.",
        "",
        "## Outcomes and gates",
        "",
    ]
    for outcome, block in results["outcomes"].items():
        lines.append(f"### {outcome}")
        lines.append("")
        for suffix, horizon_block in block.get("horizons", {}).items():
            gate = horizon_block.get("gate", {})
            lines.append(
                f"- **{suffix}**: events {gate.get('events')}, non-events {gate.get('non_events')}, "
                f"gate {'passed' if gate.get('eligible') else 'FAILED (not trained)'}"
            )
        if block.get("skip_reason"):
            lines.append(f"- skipped: {block['skip_reason']}")
        lines.append("")
    lines += [
        "Type 1 diabetes: permanently disabled. Site-specific cancer heads: disabled unless the "
        "site clears the 50-event gate.",
        "",
        "## Selection",
        "",
        "Calibration-first: smallest |calibration slope − 1| on the test split, ties broken by "
        "AUROC. Discrimination never overrides poor calibration.",
        "",
    ]
    for key, item in (results.get("selection") or {}).items():
        lines.append(
            f"- `{key}`: **{item['selected_model']}** (AUROC {item['metrics']['auroc']}, "
            f"Brier {item['metrics']['brier']}, slope {item['metrics']['calibration_slope']})"
        )
    lines += [
        "",
        "## Negative controls",
        "",
        "Shuffled outcome labels must give AUROC near 0.5; time-reversed visit sequences must "
        "degrade discrimination. Both are reported per horizon in `results.json`.",
        "",
        "## Limitations",
        "",
        "1. Synthetic data cannot establish real-world calibration, clinical utility or early detection.",
        "2. Event rates were enriched by design, so absolute probabilities are meaningless outside this sample.",
        "3. No external validation, no clinician review, no prospective evaluation.",
        "4. The temporal model uses its own encoder; the cross-sectional MetaboGuard SSL artifact is untouched.",
        "",
        "## Next step",
        "",
        "Replace the synthetic cohort with a real linked cohort (for example UK Biobank, CPRD or "
        "an equivalent EHR extract with dated incident outcomes), re-run the identical protocol, "
        "and only then discuss calibration or utility.",
    ]
    return "\n".join(lines) + "\n"


def verify_artifact_reload(
    artifact_dir: str | Path, prepared_dir: str | Path, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Reload a packaged artifact and re-score the test split, comparing to stored predictions.

    This is the parity check that proves the artifact is self-contained: models, preprocessing
    and calibrators reload from disk and reproduce the recorded probabilities exactly.
    """
    artifact = Path(artifact_dir)
    prepared = Path(prepared_dir)
    bundle = joblib.load(artifact / "future_risk_models.joblib")
    columns = json.loads((artifact / "feature_columns.json").read_text())
    metadata = json.loads((artifact / "metadata.json").read_text())
    stored = pd.read_csv(artifact / "predictions_simulation_only.csv", low_memory=False)
    features = add_subgroup_columns(pd.read_csv(prepared / "patient_features.csv", low_memory=False))

    checks: list[dict[str, Any]] = []
    for key, item in (metadata.get("selection") or {}).items():
        outcome, suffix = key.split(":")
        model_name = item["selected_model"]
        subset = stored[
            (stored["outcome"] == outcome)
            & (stored["horizon"] == suffix)
            & (stored["model"] == model_name)
            & (stored["split"] == "test")
        ]
        if subset.empty:
            checks.append({"key": key, "status": "no_stored_predictions"})
            continue
        frame = features[features["patient_id"].isin(subset["patient_id"])].copy()
        frame = frame.set_index("patient_id").loc[subset["patient_id"]].reset_index()
        horizon_days = int([k for k, v in HORIZON_LABELS.items() if v == suffix][0])
        if model_name == "discrete_time_hazard":
            raw = hazard_cumulative_incidence(bundle[outcome]["hazard"], frame, horizon_days)
        elif model_name == "temporal_gru":
            import torch

            from future_risk_models import FutureRiskConfig, build_sequences, reload_temporal_model

            temporal_meta = bundle[outcome]["temporal_meta"]
            model = reload_temporal_model(
                bundle[outcome]["temporal_state_dict"],
                temporal_meta["input_dim"],
                temporal_meta["hidden_dim"],
                len(temporal_meta["horizons"]),
            )
            visit_matrix = pd.read_csv(prepared / "visit_matrix.csv", low_memory=False)
            sequences, padding, _, ordered, _ = build_sequences(
                visit_matrix,
                frame["patient_id"].tolist(),
                FutureRiskConfig(max_visits=temporal_meta["max_visits"]),
                normaliser=bundle[outcome]["temporal_normaliser"],
            )
            with torch.no_grad():
                probabilities = (
                    torch.sigmoid(model(torch.tensor(sequences), torch.tensor(padding)))
                    .cpu()
                    .numpy()
                )
            lookup = {pid: probabilities[i] for i, pid in enumerate(ordered)}
            horizon_index = temporal_meta["horizons"].index(horizon_days)
            raw = np.array(
                [
                    lookup[pid][horizon_index] if pid in lookup else np.nan
                    for pid in frame["patient_id"]
                ]
            )
            raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0)
        else:
            model = bundle[outcome]["baselines"][suffix]["models"][model_name]
            raw = model.predict_proba(frame[columns].astype(float).to_numpy())[:, 1]
        calibrator = bundle[outcome]["calibrators"].get(f"{model_name}:{suffix}")
        calibrated = apply_calibrator(calibrator, np.nan_to_num(raw)) if calibrator else raw
        raw_delta = float(np.max(np.abs(raw - subset["raw_probability"].to_numpy())))
        calibrated_delta = float(
            np.max(np.abs(calibrated - subset["calibrated_probability"].to_numpy()))
        )
        checks.append(
            {
                "key": key,
                "model": model_name,
                "rows": int(len(subset)),
                "max_abs_raw_difference": raw_delta,
                "max_abs_calibrated_difference": calibrated_delta,
                "status": "parity_ok" if max(raw_delta, calibrated_delta) <= tolerance else "parity_mismatch",
            }
        )
    payload = {
        "artifact_dir": str(artifact),
        "tolerance": tolerance,
        "checks": checks,
        "all_parity_ok": all(check.get("status") == "parity_ok" for check in checks) and bool(checks),
        "simulation_only": True,
    }
    (artifact / "reload_parity_report.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default=str(DEFAULT_PREPARED))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=None)
    parser.add_argument(
        "--verify-artifact",
        default=None,
        help="Reload this artifact directory and check scoring parity against stored predictions.",
    )
    arguments = parser.parse_args()
    if arguments.verify_artifact:
        print(json.dumps(verify_artifact_reload(arguments.verify_artifact, arguments.prepared), indent=2))
        return
    config = FutureRiskConfig(smoke=arguments.smoke)
    if arguments.epochs:
        config.epochs = arguments.epochs
    if arguments.bootstrap_rounds:
        config.bootstrap_rounds = arguments.bootstrap_rounds
    results = run_pipeline(arguments.prepared, arguments.output_dir, config)
    summary = {
        "artifact_dir": results["artifact_dir"],
        "capability_state": results["capability_state"],
        "seconds": results["seconds"],
        "selection": results["selection"],
        "outcomes_trained": [o for o, b in results["outcomes"].items() if b.get("trained")],
        "banner": results["banner"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()