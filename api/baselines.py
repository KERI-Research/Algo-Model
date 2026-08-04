"""Unsupervised baselines for the MetaboGuard deviation score.

Two simple, fully reproducible comparators are evaluated with **exactly the same
preprocessing and split boundaries** as the self-supervised encoder:

1. **PCA reconstruction deviation** - fit PCA on the training partition, score
   reconstruction error elsewhere.
2. **Isolation Forest** - fit on the training partition, use the negated
   ``score_samples`` as a deviation score.

What is reported: reconstruction quality, deviation-score distributions,
rank agreement between methods, and flag overlap at the 95th percentile.

What is **not** reported: any disease prediction, any future-risk metric. These
baselines are unsupervised; labels are never used to fit them. Optional
cross-sectional association probes are clearly labelled as such and reuse the
same 50-case gate as the encoder.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from data_integrity import file_fingerprint, group_split_indices, validate_dataset
from self_supervised import (
    NumpyAutoencoder,
    SSLConfig,
    build_preprocessor,
    deviation_scores,
    select_prevention_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _prepare_matrix(
    frame: pd.DataFrame, config: SSLConfig
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], pd.DataFrame]:
    """Adult filter, participant-grouped split, train-only preprocessing fit."""
    features = select_prevention_features(frame)
    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18].copy()
    adult = adult.reset_index(drop=True)
    splits = group_split_indices(
        adult, fractions=tuple(config.split_fractions), seed=config.random_seed
    )
    preprocessor = build_preprocessor(features)
    preprocessor.fit(adult.iloc[splits["train"]][features])
    matrix = preprocessor.transform(adult[features]).astype(np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    keep = np.flatnonzero(finite)
    position_of = {position: index for index, position in enumerate(keep)}
    matrix = matrix[keep]
    adult = adult.iloc[keep].reset_index(drop=True)
    splits = {
        name: np.array([position_of[p] for p in indices if p in position_of], dtype=int)
        for name, indices in splits.items()
    }
    return matrix, splits, features, adult


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def run_baselines(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: SSLConfig | None = None,
    ssl_artifact_dir: str | Path | None = None,
    max_train_rows: int | None = 20_000,
    pca_components: int = 16,
    isolation_trees: int = 200,
) -> dict[str, Any]:
    """Fit and evaluate the unsupervised baselines; write a JSON report."""
    config = config or SSLConfig()
    dataset = Path(dataset_path)
    report_context = validate_dataset(dataset, strict=True)
    frame = pd.read_csv(dataset, low_memory=False)
    matrix, splits, features, adult = _prepare_matrix(frame, config)

    rng = np.random.default_rng(config.random_seed)
    train_index = splits["train"]
    if max_train_rows and len(train_index) > max_train_rows:
        train_index = np.sort(rng.choice(train_index, max_train_rows, replace=False))
    train_matrix = matrix[train_index]
    holdout_index = splits["holdout"]

    components = min(pca_components, train_matrix.shape[1])
    pca = PCA(n_components=components, random_state=config.random_seed)
    pca.fit(train_matrix)
    pca_reconstruction = pca.inverse_transform(pca.transform(matrix))
    pca_error = np.mean((pca_reconstruction - matrix) ** 2, axis=1)
    pca_reference = {
        "reconstruction_location": float(np.median(pca_error[train_index])),
        "reconstruction_scale": max(
            float(np.median(np.abs(pca_error[train_index] - np.median(pca_error[train_index]))) * 1.4826),
            1e-8,
        ),
    }
    pca_deviation = np.maximum(
        0,
        (pca_error - pca_reference["reconstruction_location"])
        / pca_reference["reconstruction_scale"],
    )

    forest = IsolationForest(
        n_estimators=isolation_trees,
        random_state=config.random_seed,
        n_jobs=1,
        contamination="auto",
    )
    forest.fit(train_matrix)
    forest_deviation = -forest.score_samples(matrix)

    results: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprint": file_fingerprint(dataset),
        "data_integrity_status": report_context.as_dict()["status"],
        "output_type": "unsupervised_deviation_baselines",
        "disclaimer": (
            "These baselines measure how unusual a metabolic profile is. They do not "
            "predict cancer or diabetes and carry no future-risk interpretation."
        ),
        "features": features,
        "transformed_dimension": int(matrix.shape[1]),
        "split_policy": {
            "grouped_by": "participant identifier (see data_integrity.identifier_column)",
            "fractions": list(config.split_fractions),
            "seed": config.random_seed,
            "fit_partition": "train",
            "sizes": {name: int(len(index)) for name, index in splits.items()},
            "train_rows_used": int(len(train_index)),
        },
        "baselines": {
            "pca_reconstruction": {
                "n_components": int(components),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "train_reconstruction_mse": float(np.mean(pca_error[train_index])),
                "validation_reconstruction_mse": float(np.mean(pca_error[splits["validation"]])),
                "holdout_reconstruction_mse": float(np.mean(pca_error[holdout_index])),
                "deviation_distribution_holdout": _distribution(pca_deviation[holdout_index]),
            },
            "isolation_forest": {
                "n_estimators": int(isolation_trees),
                "deviation_distribution_holdout": _distribution(forest_deviation[holdout_index]),
            },
        },
        "agreement": {},
    }

    holdout_scores: dict[str, np.ndarray] = {
        "pca_reconstruction": pca_deviation[holdout_index],
        "isolation_forest": forest_deviation[holdout_index],
    }

    # Optional comparison against a trained SSL artifact, using the same rows.
    if ssl_artifact_dir is not None:
        artifact = Path(ssl_artifact_dir)
        metadata = json.loads((artifact / "metadata.json").read_text())
        if metadata["features"] != features:
            results["ssl_comparison_skipped"] = (
                "Artifact feature list differs from the current allowlist."
            )
        else:
            import joblib

            artifact_preprocessor = joblib.load(artifact / "preprocessor.joblib")
            artifact_matrix = artifact_preprocessor.transform(adult[features]).astype(np.float32)
            scorer = NumpyAutoencoder(artifact / "autoencoder_weights.npz")
            reconstructed, latent = scorer.reconstruct(artifact_matrix)
            reconstruction_error = np.mean((reconstructed - artifact_matrix) ** 2, axis=1)
            distribution = metadata["score_distribution"]
            latent_distance = np.mean(
                (
                    (latent - np.asarray(distribution["latent_mean"]))
                    / np.asarray(distribution["latent_std"])
                )
                ** 2,
                axis=1,
            )
            ssl_deviation = deviation_scores(reconstruction_error, latent_distance, distribution)
            holdout_scores["metaboguard_ssl"] = ssl_deviation[holdout_index]
            results["baselines"]["metaboguard_ssl"] = {
                "artifact_dir": str(artifact),
                "run_label": metadata.get("run_label"),
                "code_version": metadata.get("code_version"),
                "holdout_reconstruction_mse": float(np.mean(reconstruction_error[holdout_index])),
                "deviation_distribution_holdout": _distribution(ssl_deviation[holdout_index]),
            }

    names = sorted(holdout_scores)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            correlation = float(spearmanr(holdout_scores[first], holdout_scores[second]).statistic)
            threshold_first = np.quantile(holdout_scores[first], 0.95)
            threshold_second = np.quantile(holdout_scores[second], 0.95)
            flag_first = holdout_scores[first] >= threshold_first
            flag_second = holdout_scores[second] >= threshold_second
            union = int((flag_first | flag_second).sum())
            results["agreement"][f"{first}__vs__{second}"] = {
                "spearman_rank_correlation_holdout": correlation,
                "top5pct_flag_jaccard": float(
                    int((flag_first & flag_second).sum()) / union if union else 0.0
                ),
            }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline_report.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "model_artifacts" / "benchmarks"),
    )
    parser.add_argument("--ssl-artifact", default=None, help="Optional trained SSL artifact directory.")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--isolation-trees", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=20_000)
    arguments = parser.parse_args()

    results = run_baselines(
        arguments.dataset,
        arguments.output_dir,
        ssl_artifact_dir=arguments.ssl_artifact,
        max_train_rows=arguments.max_train_rows,
        pca_components=arguments.pca_components,
        isolation_trees=arguments.isolation_trees,
    )
    print(
        json.dumps(
            {
                "output_dir": arguments.output_dir,
                "baselines": {
                    name: {
                        key: value
                        for key, value in payload.items()
                        if key != "deviation_distribution_holdout"
                    }
                    for name, payload in results["baselines"].items()
                },
                "agreement": results["agreement"],
                "disclaimer": results["disclaimer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()