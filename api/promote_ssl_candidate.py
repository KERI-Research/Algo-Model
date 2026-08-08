#!/usr/bin/env python3
"""Promote an SSL artifact only after objective and stability gates pass."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_integrity import file_fingerprint, group_split_indices
from self_supervised import NumpyAutoencoder, deviation_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = PROJECT_ROOT / "model_artifacts" / "metaboguard_ssl"
POINTER_PATH = ARTIFACT_ROOT / "CURRENT.json"


def _metadata(artifact: Path) -> dict[str, Any]:
    path = artifact / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Artifact metadata is missing: {path}")
    return json.loads(path.read_text())


def _minimum_validation_loss(metadata: dict[str, Any]) -> float:
    history = metadata.get("training_history") or []
    values = [float(row["validation_loss"]) for row in history]
    if not values:
        raise ValueError("Artifact has no validation-loss history.")
    return min(values)


def _holdout_scores(
    artifact: Path,
    metadata: dict[str, Any],
    adult: pd.DataFrame,
    holdout: np.ndarray,
) -> tuple[float, np.ndarray]:
    features = metadata["features"]
    preprocessor = joblib.load(artifact / "preprocessor.joblib")
    matrix = preprocessor.transform(adult[features]).astype(np.float32)
    scorer = NumpyAutoencoder(artifact / "autoencoder_weights.npz")
    reconstructed, latent = scorer.reconstruct(matrix)
    reconstruction_error = np.mean((reconstructed - matrix) ** 2, axis=1)
    distribution = metadata["score_distribution"]
    latent_distance = np.mean(
        (
            (latent - np.asarray(distribution["latent_mean"]))
            / np.asarray(distribution["latent_std"])
        )
        ** 2,
        axis=1,
    )
    scores = deviation_scores(reconstruction_error, latent_distance, distribution)
    return float(np.mean(reconstruction_error[holdout])), scores[holdout]


def evaluate_candidate(
    candidate: Path,
    baseline: Path,
    dataset: Path,
    minimum_mse_improvement: float = 0.01,
    minimum_rank_correlation: float = 0.80,
) -> dict[str, Any]:
    candidate_metadata = _metadata(candidate)
    baseline_metadata = _metadata(baseline)
    if candidate_metadata.get("run_label") == "smoke":
        raise ValueError("A smoke artifact cannot be promoted.")
    for key in ("features", "latent_dim", "output_type"):
        if candidate_metadata.get(key) != baseline_metadata.get(key):
            raise ValueError(f"Candidate changes the public artifact contract: {key}.")

    fingerprint = file_fingerprint(dataset)
    candidate_fingerprint = candidate_metadata.get("dataset_fingerprint") or {}
    if candidate_fingerprint.get("sha256") != fingerprint["sha256"]:
        raise ValueError("Candidate dataset fingerprint does not match the current dataset.")

    frame = pd.read_csv(dataset, low_memory=False)
    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18]
    adult = adult.reset_index(drop=True)
    split = group_split_indices(adult, fractions=(0.7, 0.15, 0.15), seed=42)
    holdout = split["holdout"]

    baseline_mse, baseline_scores = _holdout_scores(
        baseline, baseline_metadata, adult, holdout
    )
    candidate_mse, candidate_scores = _holdout_scores(
        candidate, candidate_metadata, adult, holdout
    )
    improvement = (baseline_mse - candidate_mse) / baseline_mse
    rank_correlation = float(
        spearmanr(baseline_scores, candidate_scores).statistic
    )
    baseline_flags = baseline_scores >= np.quantile(baseline_scores, 0.95)
    candidate_flags = candidate_scores >= np.quantile(candidate_scores, 0.95)
    union = int((baseline_flags | candidate_flags).sum())
    top_five_jaccard = float(
        int((baseline_flags & candidate_flags).sum()) / union if union else 1.0
    )
    baseline_validation = _minimum_validation_loss(baseline_metadata)
    candidate_validation = _minimum_validation_loss(candidate_metadata)

    gates = {
        "validation_improved": candidate_validation < baseline_validation,
        "holdout_mse_improved": improvement >= minimum_mse_improvement,
        "deviation_rank_stable": rank_correlation >= minimum_rank_correlation,
        "same_dataset": True,
        "same_public_contract": True,
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_objective": "label-free reconstruction; disease labels excluded",
        "candidate": candidate.name,
        "baseline": baseline.name,
        "dataset": fingerprint["name"],
        "dataset_sha256": fingerprint["sha256"],
        "metrics": {
            "baseline_min_validation_loss": baseline_validation,
            "candidate_min_validation_loss": candidate_validation,
            "baseline_holdout_reconstruction_mse": baseline_mse,
            "candidate_holdout_reconstruction_mse": candidate_mse,
            "holdout_mse_improvement_fraction": improvement,
            "deviation_spearman_holdout": rank_correlation,
            "top5_flag_jaccard_holdout": top_five_jaccard,
        },
        "thresholds": {
            "minimum_mse_improvement_fraction": minimum_mse_improvement,
            "minimum_deviation_rank_correlation": minimum_rank_correlation,
        },
        "gates": gates,
        "verdict": "promotable" if all(gates.values()) else "rejected",
        "scope": (
            "Metabolic deviation and representation only. This comparison does not "
            "measure diagnosis or future disease prediction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv",
    )
    parser.add_argument("--minimum-mse-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-rank-correlation", type=float, default=0.80)
    parser.add_argument("--promote", action="store_true")
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    baseline = arguments.baseline.resolve()
    report = evaluate_candidate(
        candidate,
        baseline,
        arguments.dataset.resolve(),
        arguments.minimum_mse_improvement,
        arguments.minimum_rank_correlation,
    )
    (candidate / "promotion_report.json").write_text(json.dumps(report, indent=2))

    if arguments.promote:
        if report["verdict"] != "promotable":
            raise SystemExit("Promotion refused: one or more gates failed.")
        POINTER_PATH.write_text(
            json.dumps(
                {
                    "artifact_dir": str(candidate),
                    "run_label": _metadata(candidate).get("run_label"),
                    "promoted_at": datetime.now(UTC).isoformat(),
                    "promotion_report": str(candidate / "promotion_report.json"),
                    "output_type": _metadata(candidate).get("output_type"),
                },
                indent=2,
            )
        )
        report["promoted_to"] = str(POINTER_PATH)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()