"""Label-free phenotype clustering over the frozen MetaboGuard representation.

Following Prof. Nada's recommendation (see
``docs/decisions/2026-08-04-professor-feedback.md``), this module looks for **patient /
metabolic phenotypes** without using any disease label to fit or to select a solution.

Hard rules implemented here:

* **No labels in fit or selection.** Labels are read only after a solution is chosen, and
  only to report cross-sectional prevalence, suppressed at small counts.
* **Clusters are phenotypes, never diagnoses.** No cluster may be named after a cancer,
  a cancer site or a subtype; the output schema has no field for one.
* **Frozen encoder.** The self-supervised artifact is loaded read-only; clustering never
  retrains it. Preprocessing comes from the artifact, so it stays fit-on-train-only.
* **Persisted split boundaries.** ``splits.npz`` from the artifact is reused when it
  matches the current row count; otherwise the same seeded, participant-grouped policy is
  recomputed and that fact is recorded.
* **Negative controls are mandatory.** A solution whose clusters are explained by survey
  cycle, assay availability, missingness burden, age or sex is reported as a data
  artefact, not a phenotype.
* **Abstain is a valid result.** When no candidate passes the quality, stability and
  negative-control gates the report status is ``no_stable_clusters``.

CLI::

    python clustering.py --artifact ../model_artifacts/metaboguard_ssl/meeting_2026-08-04/ssl_artifact \
        --output-dir ../model_artifacts/clustering/run1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

try:  # HDBSCAN is in-tree from scikit-learn 1.3; no extra dependency is added.
    from sklearn.cluster import HDBSCAN

    HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed sklearn
    HDBSCAN = None
    HDBSCAN_AVAILABLE = False
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

import data_integrity as di
from self_supervised import NumpyAutoencoder, deviation_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_VERSION = "metaboguard-clustering-v1"

#: Minimum cases per class before any post-hoc prevalence figure is reported.
MIN_POSTHOC_CASES = 50


@dataclass
class ClusterConfig:
    #: "latent" = frozen 16-d SSL representation; "features" = shared preprocessed matrix.
    space: str = "latent"
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    #: HDBSCAN is preferred for the density arm and degrades to DBSCAN, then to
    #: "unavailable", without adding dependencies.
    methods: tuple[str, ...] = ("kmeans", "gaussian_mixture", "hdbscan")
    random_seed: int = 42
    #: Extra seeds used only to measure seed stability.
    stability_seeds: tuple[int, ...] = (43, 44)
    bootstrap_rounds: int = 20
    bootstrap_fraction: float = 0.8
    max_fit_rows: int = 20_000
    silhouette_sample: int = 5_000
    #: Quality / stability gates.
    min_silhouette: float = 0.15
    min_cluster_fraction: float = 0.05
    min_bootstrap_ari: float = 0.60
    min_seed_ari: float = 0.60
    #: A cluster solution explained this strongly by a nuisance variable is an artefact.
    max_negative_control_association: float = 0.30
    dbscan_min_samples: int = 25
    #: Cluster-wise bootstrap Jaccard thresholds (Hennig): <=0.5 means the cluster
    #: dissolves under resampling, >0.75 means good recovery.
    jaccard_dissolution_threshold: float = 0.50
    jaccard_recovery_threshold: float = 0.75
    #: Column-permutation null: destroys between-feature structure while keeping margins.
    permutation_rounds: int = 3
    #: A solution must beat the permuted-null silhouette by at least this margin.
    min_silhouette_gain_over_null: float = 0.05
    #: Top-1% deviation-score outliers are dropped to test whether they drive the solution.
    outlier_sensitivity_fraction: float = 0.01
    min_outlier_sensitivity_ari: float = 0.60
    projection_sample: int = 2_000
    #: Restrict rows to complete cases on the reliability-tier `usable_now` features.
    #: This is the sensitivity analysis that removes missingness/assay-availability
    #: patterns as a possible driver of the clustering.
    restrict_to_complete_cases: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("k_values", "methods", "stability_seeds"):
            payload[key] = list(getattr(self, key))
        return payload


# ---------------------------------------------------------------------------
# Association helpers (used for negative controls and characterisation)
# ---------------------------------------------------------------------------


def cramers_v(
    labels: np.ndarray, categories: np.ndarray, bias_correction: bool = True
) -> float:
    """Cramér's V between cluster labels and a categorical nuisance variable.

    The uncorrected statistic is biased upward when the nuisance variable has many
    categories, which would make any high-cardinality control (for example a per-row
    assay-availability pattern) look dominant regardless of the clustering. The
    Bergsma bias correction is therefore applied by default, and the raw value is
    available with ``bias_correction=False`` for transparency.
    """
    from scipy.stats import chi2_contingency

    table = pd.crosstab(pd.Series(labels), pd.Series(categories))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    counts = table.to_numpy()
    n = counts.sum()
    if n == 0:
        return 0.0
    chi2 = chi2_contingency(counts)[0]
    rows, columns = counts.shape
    if not bias_correction:
        denominator = n * (min(rows, columns) - 1)
        return float(np.sqrt(chi2 / denominator)) if denominator > 0 else 0.0
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1))
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    columns_corrected = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))


def correlation_ratio(labels: np.ndarray, values: np.ndarray) -> float:
    """Eta: how much of a continuous variable's variance the clustering explains."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 10:
        return 0.0
    values = values[finite]
    labels = np.asarray(labels)[finite]
    grand_mean = values.mean()
    total = float(((values - grand_mean) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        if group.size:
            between += group.size * (group.mean() - grand_mean) ** 2
    return float(np.sqrt(max(between, 0.0) / total))


def _robust_smd(group: np.ndarray, reference: np.ndarray) -> float:
    """Robust standardised difference: (median_group - median_ref) / reference MAD."""
    group = group[np.isfinite(group)]
    reference = reference[np.isfinite(reference)]
    if group.size < 5 or reference.size < 5:
        return float("nan")
    reference_median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - reference_median))) * 1.4826
    if mad <= 0:
        return float("nan")
    return float((np.median(group) - reference_median) / mad)


# ---------------------------------------------------------------------------
# Data preparation from a frozen artifact
# ---------------------------------------------------------------------------


@dataclass
class ClusteringInputs:
    frame: pd.DataFrame  # adult, finite rows, aligned with matrices
    latent: np.ndarray
    features_matrix: np.ndarray
    feature_names: list[str]
    raw_features: list[str]
    splits: dict[str, np.ndarray]
    split_source: str
    deviation: np.ndarray
    artifact_metadata: dict[str, Any]


def usable_now_features(dataset_path: str | Path) -> list[str]:
    """Features the reliability report rates `usable_now` for this dataset."""
    from data_reliability import build_reliability_report

    report = build_reliability_report(dataset_path, strict=False)
    return report.tier("usable_now")


def prepare_inputs(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    config: ClusterConfig | None = None,
) -> ClusteringInputs:
    """Load the dataset through the frozen artifact's preprocessing and encoder."""
    dataset = Path(dataset_path)
    di.assert_dataset_allowed(dataset)
    artifact = Path(artifact_dir)
    metadata = json.loads((artifact / "metadata.json").read_text())
    preprocessor = joblib.load(artifact / "preprocessor.joblib")
    raw_features: list[str] = metadata["features"]

    frame = pd.read_csv(dataset, low_memory=False)
    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18].reset_index(
        drop=True
    )
    complete_case_note: str | None = None
    if config is not None and config.restrict_to_complete_cases:
        usable = [
            column for column in usable_now_features(dataset) if column in adult.columns
        ]
        before = len(adult)
        adult = adult[adult[usable].notna().all(axis=1)].reset_index(drop=True)
        complete_case_note = (
            f"Restricted to complete cases on {len(usable)} usable_now features: "
            f"{len(adult)} of {before} adult rows retained. This removes missingness and "
            "assay-availability patterns as possible drivers of the clustering."
        )

    matrix = preprocessor.transform(adult[raw_features]).astype(np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    keep = np.flatnonzero(finite)
    matrix = matrix[keep]
    adult = adult.iloc[keep].reset_index(drop=True)

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
    deviation = deviation_scores(reconstruction_error, latent_distance, distribution)

    splits_path = artifact / "splits.npz"
    if complete_case_note is not None:
        # Persisted indices refer to the full adult frame, so they cannot be reused here.
        splits = di.group_split_indices(adult, seed=metadata["config"]["random_seed"])
        split_source = "recomputed:complete_case_subset - " + complete_case_note
    elif splits_path.exists():
        archive = np.load(splits_path)
        total = sum(len(archive[name]) for name in ("train", "validation", "holdout"))
        if total == len(adult):
            splits = {
                name: np.asarray(archive[name], dtype=int)
                for name in ("train", "validation", "holdout")
            }
            split_source = f"persisted:{splits_path.name}"
        else:
            splits = di.group_split_indices(adult, seed=metadata["config"]["random_seed"])
            split_source = (
                "recomputed:persisted split row count "
                f"({total}) did not match current rows ({len(adult)})"
            )
    else:
        splits = di.group_split_indices(adult, seed=metadata["config"]["random_seed"])
        split_source = "recomputed:no splits.npz in artifact"

    return ClusteringInputs(
        frame=adult,
        latent=latent,
        features_matrix=matrix,
        feature_names=metadata["transformed_feature_names"],
        raw_features=raw_features,
        splits=splits,
        split_source=split_source,
        deviation=deviation,
        artifact_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Candidate fitting
# ---------------------------------------------------------------------------


def _fit(method: str, k: int, matrix: np.ndarray, config: ClusterConfig, seed: int):
    if method == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matrix)
    if method == "gaussian_mixture":
        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            reg_covar=1e-4,
            max_iter=300,
        ).fit(matrix)
    if method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise RuntimeError(
                "HDBSCAN is unavailable in this scikit-learn build. The density arm degrades "
                "to DBSCAN (method='dbscan') rather than installing a new dependency."
            )
        # k parameterises the minimum cluster size as a fraction of the fitted rows, so the
        # density arm explores granularity the same way the centroid methods explore k.
        minimum = max(config.dbscan_min_samples, int(len(matrix) * config.min_cluster_fraction / k))
        return HDBSCAN(
            min_cluster_size=minimum,
            min_samples=config.dbscan_min_samples,
            allow_single_cluster=False,
        ).fit(matrix)
    if method == "dbscan":
        # eps derived from the data scale so it is deterministic and reported.
        distances = np.linalg.norm(
            matrix[: min(len(matrix), 2000)] - matrix[: min(len(matrix), 2000)].mean(axis=0),
            axis=1,
        )
        eps = float(np.quantile(distances, k / 10.0))
        return DBSCAN(eps=max(eps, 1e-3), min_samples=config.dbscan_min_samples).fit(matrix)
    raise ValueError(f"Unknown method '{method}'")


def _permutation_null(
    method: str, k: int, matrix: np.ndarray, config: ClusterConfig
) -> dict[str, Any]:
    """Column-permutation null: shuffle each column independently and refit.

    Permuting columns preserves every marginal distribution but destroys the joint
    structure, so any silhouette the method still achieves is geometry it would find in
    structureless data of the same shape. A solution must beat this null to count.
    """
    rng = np.random.default_rng(config.random_seed + 101)
    scores: list[float] = []
    for _ in range(config.permutation_rounds):
        permuted = np.column_stack(
            [rng.permutation(matrix[:, column]) for column in range(matrix.shape[1])]
        )
        try:
            model = _fit(method, k, permuted, config, config.random_seed)
            labels = _labels(model, permuted)
            metrics = _internal_metrics(permuted, labels, config)
        except Exception:
            continue
        if metrics.get("silhouette") is not None:
            scores.append(float(metrics["silhouette"]))
    if not scores:
        return {"rounds": 0, "mean_null_silhouette": None}
    return {
        "rounds": len(scores),
        "mean_null_silhouette": round(float(np.mean(scores)), 4),
        "max_null_silhouette": round(float(np.max(scores)), 4),
        "definition": "Each column permuted independently; marginals kept, joint structure destroyed.",
    }


def _outlier_sensitivity(
    method: str,
    k: int,
    matrix: np.ndarray,
    reference_labels: np.ndarray,
    deviation: np.ndarray,
    config: ClusterConfig,
) -> dict[str, Any]:
    """Refit without the most extreme rows: a solution driven by outliers is fragile."""
    count = max(1, int(len(matrix) * config.outlier_sensitivity_fraction))
    drop = np.argsort(deviation)[-count:]
    keep = np.setdiff1d(np.arange(len(matrix)), drop)
    try:
        model = _fit(method, k, matrix[keep], config, config.random_seed)
        labels = _labels(model, matrix[keep])
    except Exception as error:  # pragma: no cover
        return {"status": "failed", "error": str(error)[:150]}
    return {
        "status": "evaluated",
        "dropped_rows": int(count),
        "dropped_fraction": config.outlier_sensitivity_fraction,
        "ari_vs_full_fit": round(
            float(adjusted_rand_score(reference_labels[keep], labels)), 4
        ),
        "definition": (
            "Top deviation-score rows removed, then refit; ARI against the full-data labels "
            "on the retained rows."
        ),
    }


def _labels(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(matrix))
    return np.asarray(model.labels_)


def _internal_metrics(
    matrix: np.ndarray, labels: np.ndarray, config: ClusterConfig
) -> dict[str, Any]:
    unique = [label for label in np.unique(labels) if label != -1]
    if len(unique) < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "reason": "fewer than two clusters",
        }
    mask = labels != -1
    subset_matrix, subset_labels = matrix[mask], labels[mask]
    rng = np.random.default_rng(config.random_seed)
    if len(subset_matrix) > config.silhouette_sample:
        index = rng.choice(len(subset_matrix), config.silhouette_sample, replace=False)
        sample_matrix, sample_labels = subset_matrix[index], subset_labels[index]
    else:
        sample_matrix, sample_labels = subset_matrix, subset_labels
    if len(np.unique(sample_labels)) < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "reason": "subsample collapsed to one cluster",
        }
    return {
        "silhouette": round(float(silhouette_score(sample_matrix, sample_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(subset_matrix, subset_labels)), 4),
        "calinski_harabasz": round(
            float(calinski_harabasz_score(subset_matrix, subset_labels)), 2
        ),
        "silhouette_sample_size": int(len(sample_matrix)),
    }


def _size_profile(labels: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    sizes = {
        ("noise" if int(value) == -1 else f"cluster_{int(value)}"): int(count)
        for value, count in zip(values, counts)
    }
    cluster_counts = [
        int(count) for value, count in zip(values, counts) if int(value) != -1
    ]
    return {
        "sizes": sizes,
        "n_clusters": len(cluster_counts),
        "smallest_cluster_fraction": round(
            float(min(cluster_counts) / total) if cluster_counts else 0.0, 6
        ),
        "noise_fraction": round(float(sizes.get("noise", 0) / total), 6),
    }


def _bootstrap_stability(
    method: str, k: int, matrix: np.ndarray, reference_labels: np.ndarray, config: ClusterConfig
) -> dict[str, Any]:
    """Resample-refit stability: adjusted Rand index against the reference labelling."""
    rng = np.random.default_rng(config.random_seed + 7)
    scores: list[float] = []
    reference_clusters = [label for label in np.unique(reference_labels) if label != -1]
    jaccard: dict[int, list[float]] = {label: [] for label in reference_clusters}
    size = int(len(matrix) * config.bootstrap_fraction)
    for _ in range(config.bootstrap_rounds):
        index = rng.choice(len(matrix), size, replace=False)
        try:
            model = _fit(method, k, matrix[index], config, config.random_seed)
            labels = _labels(model, matrix[index])
        except Exception:
            continue
        scores.append(float(adjusted_rand_score(reference_labels[index], labels)))
        # Cluster-wise Jaccard (Hennig): for each original cluster, the best overlap with
        # any cluster in the resampled solution.
        for label in reference_clusters:
            original = reference_labels[index] == label
            if not original.any():
                continue
            best = 0.0
            for candidate in np.unique(labels):
                if candidate == -1:
                    continue
                new = labels == candidate
                union = int((original | new).sum())
                if union:
                    best = max(best, float((original & new).sum()) / union)
            jaccard[label].append(best)
    if not scores:
        return {"mean_ari": None, "sd_ari": None, "rounds": 0, "clusterwise_jaccard": {}}
    per_cluster = {
        f"cluster_{int(label)}": round(float(np.mean(values)), 4)
        for label, values in jaccard.items()
        if values
    }
    dissolved = sorted(
        name
        for name, value in per_cluster.items()
        if value <= config.jaccard_dissolution_threshold
    )
    recovered = sorted(
        name
        for name, value in per_cluster.items()
        if value > config.jaccard_recovery_threshold
    )
    return {
        "mean_ari": round(float(np.mean(scores)), 4),
        "sd_ari": round(float(np.std(scores)), 4),
        "min_ari": round(float(np.min(scores)), 4),
        "rounds": len(scores),
        "clusterwise_jaccard": per_cluster,
        "min_clusterwise_jaccard": (
            round(float(min(per_cluster.values())), 4) if per_cluster else None
        ),
        "dissolved_clusters": dissolved,
        "well_recovered_clusters": recovered,
        "thresholds": {
            "dissolution_at_or_below": config.jaccard_dissolution_threshold,
            "good_recovery_above": config.jaccard_recovery_threshold,
        },
        "note": (
            "Cluster-wise Jaccard measures whether each individual cluster survives "
            "resampling; a good global ARI can still hide one dissolving cluster. "
            "Stability alone does not establish validity."
        ),
    }


def _seed_stability(
    method: str, k: int, matrix: np.ndarray, reference_labels: np.ndarray, config: ClusterConfig
) -> dict[str, Any]:
    scores: list[float] = []
    for seed in config.stability_seeds:
        try:
            model = _fit(method, k, matrix, config, seed)
            scores.append(float(adjusted_rand_score(reference_labels, _labels(model, matrix))))
        except Exception:
            continue
    if not scores:
        return {"mean_ari": None, "seeds": list(config.stability_seeds)}
    return {
        "mean_ari": round(float(np.mean(scores)), 4),
        "min_ari": round(float(np.min(scores)), 4),
        "seeds": list(config.stability_seeds),
    }


def _negative_controls(
    labels: np.ndarray, frame: pd.DataFrame, raw_features: list[str], config: ClusterConfig
) -> dict[str, Any]:
    """Flag clusters that are explained by the survey rather than by biology."""
    mask = labels != -1
    labels = labels[mask]
    frame = frame.iloc[np.flatnonzero(mask)]
    controls: dict[str, Any] = {}

    if "survey_cycle" in frame.columns:
        controls["survey_cycle"] = {
            "statistic": "cramers_v_bias_corrected",
            "value": round(cramers_v(labels, frame["survey_cycle"].astype(str).to_numpy()), 4),
        }
    available = [column for column in raw_features if column in frame.columns]
    if available:
        missing_count = frame[available].isna().sum(axis=1).to_numpy()
        controls["missingness_burden"] = {
            "statistic": "correlation_ratio",
            "value": round(correlation_ratio(labels, missing_count), 4),
        }
        # Gating control: how many lab features a participant had measured, binned into a
        # small number of categories. Low cardinality keeps the statistic interpretable.
        observed_count = frame[available].notna().sum(axis=1).to_numpy()
        bins = np.digitize(
            observed_count,
            np.unique(np.quantile(observed_count, [0.25, 0.5, 0.75])),
        )
        controls["assay_availability_burden"] = {
            "statistic": "cramers_v_bias_corrected",
            "value": round(cramers_v(labels, bins.astype(str)), 4),
            "definition": "Quartile bins of the number of observed model-input features.",
        }
        # Secondary diagnostic only, not gating: the exact per-row availability pattern is
        # high-cardinality, so even the bias-corrected statistic is hard to interpret.
        assay_pattern = (
            frame[available].notna().astype(int).astype(str).agg("".join, axis=1).to_numpy()
        )
        controls["assay_availability_pattern_diagnostic"] = {
            "statistic": "cramers_v_bias_corrected",
            "value": round(cramers_v(labels, assay_pattern), 4),
            "gating": False,
            "note": (
                "High-cardinality diagnostic. Reported for transparency; the gating decision "
                "uses assay_availability_burden."
            ),
        }
    if "DEMO_RIDAGEYR" in frame.columns:
        controls["age"] = {
            "statistic": "correlation_ratio",
            "value": round(
                correlation_ratio(
                    labels, pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce").to_numpy()
                ),
                4,
            ),
        }
    if "DEMO_RIAGENDR" in frame.columns:
        controls["sex"] = {
            "statistic": "cramers_v_bias_corrected",
            "value": round(cramers_v(labels, frame["DEMO_RIAGENDR"].astype(str).to_numpy()), 4),
        }

    dominated_by = sorted(
        name
        for name, payload in controls.items()
        if payload.get("gating", True)
        and payload["value"] is not None
        and payload["value"] > config.max_negative_control_association
    )
    return {
        "threshold": config.max_negative_control_association,
        "controls": controls,
        "dominated_by": dominated_by,
        "is_data_artefact": bool(dominated_by),
        "interpretation": (
            "A high value means the clustering largely reproduces that nuisance variable. "
            "Age and sex associations are expected to be non-zero in metabolic data; only "
            "values above the threshold mark the solution as a data artefact."
        ),
    }


# ---------------------------------------------------------------------------
# Characterisation
# ---------------------------------------------------------------------------


def _membership_confidence(model, matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        top = probabilities.max(axis=1)
        return {
            "method": "gaussian_mixture_posterior",
            "mean_top_probability": round(float(top.mean()), 4),
            "fraction_confident_above_0_8": round(float((top >= 0.8).mean()), 4),
            "per_cluster_mean_probability": {
                f"cluster_{int(label)}": round(float(top[labels == label].mean()), 4)
                for label in np.unique(labels)
                if label != -1
            },
        }
    if hasattr(model, "transform"):
        distances = model.transform(matrix)
        ordered = np.sort(distances, axis=1)
        margin = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered[:, 1], 1e-9)
        return {
            "method": "kmeans_relative_margin",
            "definition": "(d2 - d1) / d2 using distances to the two nearest centroids",
            "mean_margin": round(float(margin.mean()), 4),
            "fraction_margin_above_0_2": round(float((margin >= 0.2).mean()), 4),
            "per_cluster_mean_margin": {
                f"cluster_{int(label)}": round(float(margin[labels == label].mean()), 4)
                for label in np.unique(labels)
                if label != -1
            },
        }
    return {"method": "not_available", "reason": "method exposes no confidence measure"}


def _posthoc_prevalence(frame: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    """Cross-sectional prevalence per cluster. Suppressed at small counts."""
    result: dict[str, Any] = {
        "explanation_class": "model_association",
        "output_type": "cross_sectional_association_only",
        "warning": (
            "Prevalence of an ALREADY-recorded diagnosis inside each phenotype. This is not "
            "future risk, not a diagnosis and not evidence that the phenotype causes disease."
        ),
        "labels": {},
    }
    for column in ("Cancer", "Diabetes"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        positives = int(np.nansum(values == 1))
        negatives = int(np.nansum(values == 0))
        if positives < MIN_POSTHOC_CASES or negatives < MIN_POSTHOC_CASES:
            result["labels"][column] = {
                "status": "suppressed",
                "reason": f"Fewer than {MIN_POSTHOC_CASES} cases per class overall.",
                "positives": positives,
                "negatives": negatives,
            }
            continue
        per_cluster: dict[str, Any] = {}
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_values = values[mask]
            cluster_positives = int(np.nansum(cluster_values == 1))
            cluster_known = int(np.sum(np.isfinite(cluster_values)))
            if cluster_positives < MIN_POSTHOC_CASES:
                per_cluster[f"cluster_{int(label)}"] = {
                    "status": "suppressed",
                    "reason": f"Fewer than {MIN_POSTHOC_CASES} recorded cases in this phenotype.",
                    "rows": int(mask.sum()),
                }
                continue
            per_cluster[f"cluster_{int(label)}"] = {
                "status": "reported",
                "rows": int(mask.sum()),
                "known_label_rows": cluster_known,
                "recorded_cases": cluster_positives,
                "prevalence": round(float(cluster_positives / max(cluster_known, 1)), 6),
            }
        result["labels"][column] = {
            "status": "reported",
            "overall_prevalence": round(float(positives / max(positives + negatives, 1)), 6),
            "per_cluster": per_cluster,
        }
    # Pancreatic cancer is always suppressed: 19 prevalent cases repo-wide.
    if "PancreaticCancer" in frame.columns:
        positives = int(pd.to_numeric(frame["PancreaticCancer"], errors="coerce").eq(1).sum())
        result["labels"]["PancreaticCancer"] = {
            "status": "suppressed",
            "reason": (
                f"{positives} prevalent cases in this partition; far below the "
                f"{MIN_POSTHOC_CASES}-case reporting floor. No site-level output is produced."
            ),
        }
    return result


def characterise(
    inputs: ClusteringInputs,
    labels: np.ndarray,
    model,
    matrix: np.ndarray,
    partition_index: np.ndarray,
    config: ClusterConfig,
) -> dict[str, Any]:
    frame = inputs.frame.iloc[partition_index].reset_index(drop=True)
    numeric_features = [
        column
        for column in inputs.raw_features
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    reference = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy()
        for column in numeric_features
    }
    deviation = inputs.deviation[partition_index]

    clusters: list[dict[str, Any]] = []
    for label in np.unique(labels):
        if label == -1:
            continue
        mask = labels == label
        prototype: dict[str, float | None] = {}
        directions: list[dict[str, Any]] = []
        for column in numeric_features:
            values = reference[column][mask]
            observed = values[np.isfinite(values)]
            prototype[column] = round(float(np.median(observed)), 4) if observed.size else None
            smd = _robust_smd(values, reference[column])
            if np.isfinite(smd):
                directions.append(
                    {
                        "feature": column,
                        "robust_standardised_difference": round(float(smd), 4),
                        "direction": "higher" if smd > 0 else "lower",
                    }
                )
        directions.sort(key=lambda item: abs(item["robust_standardised_difference"]), reverse=True)
        clusters.append(
            {
                "cluster_id": f"cluster_{int(label)}",
                "label_policy": (
                    "Phenotype identifier only. Naming this after a disease, cancer type or "
                    "cancer site is prohibited."
                ),
                "rows": int(mask.sum()),
                "share_of_partition": round(float(mask.mean()), 6),
                "prototype_median_profile": prototype,
                "top_distinguishing_panel": directions[:8],
                "deviation_score_summary": {
                    "median": round(float(np.median(deviation[mask])), 6),
                    "p90": round(float(np.quantile(deviation[mask], 0.9)), 6),
                },
            }
        )
    return {
        "clusters": clusters,
        "membership_confidence": _membership_confidence(model, matrix, labels),
        "posthoc_label_summary": _posthoc_prevalence(frame, labels),
        "panel_framing": (
            "Distinguishing features are reported as panels of interacting risk-associated "
            "features, consistent with the meeting conclusion that early detection needs "
            "panels rather than one universal marker."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_clustering(
    dataset_path: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    config: ClusterConfig | None = None,
) -> dict[str, Any]:
    """Fit, evaluate, gate and characterise candidate cluster solutions."""
    config = config or ClusterConfig()
    inputs = prepare_inputs(dataset_path, artifact_dir, config)
    matrix_all = inputs.latent if config.space == "latent" else inputs.features_matrix

    rng = np.random.default_rng(config.random_seed)
    train_index = inputs.splits["train"]
    if len(train_index) > config.max_fit_rows:
        train_index = np.sort(rng.choice(train_index, config.max_fit_rows, replace=False))
    holdout_index = inputs.splits["holdout"]
    train_matrix = matrix_all[train_index]
    holdout_matrix = matrix_all[holdout_index]

    candidates: list[dict[str, Any]] = []
    for method in config.methods:
        for k in config.k_values:
            try:
                model = _fit(method, k, train_matrix, config, config.random_seed)
                train_labels = _labels(model, train_matrix)
            except Exception as error:  # pragma: no cover - method/parameter failure
                candidates.append(
                    {
                        "method": method,
                        "k": k,
                        "status": "failed",
                        "error": str(error)[:200],
                    }
                )
                continue
            profile = _size_profile(train_labels)
            entry: dict[str, Any] = {
                "method": method,
                "k": k,
                "status": "evaluated",
                "size_profile": profile,
                "train_metrics": _internal_metrics(train_matrix, train_labels, config),
                "bootstrap_stability": _bootstrap_stability(
                    method, k, train_matrix, train_labels, config
                ),
                "seed_stability": _seed_stability(
                    method, k, train_matrix, train_labels, config
                ),
                "negative_controls": _negative_controls(
                    train_labels, inputs.frame.iloc[train_index], inputs.raw_features, config
                ),
                "permutation_null": _permutation_null(method, k, train_matrix, config),
                "outlier_sensitivity": _outlier_sensitivity(
                    method,
                    k,
                    train_matrix,
                    train_labels,
                    inputs.deviation[train_index],
                    config,
                ),
            }
            if hasattr(model, "predict"):
                holdout_labels = _labels(model, holdout_matrix)
                entry["holdout_metrics"] = _internal_metrics(
                    holdout_matrix, holdout_labels, config
                )
                entry["holdout_size_profile"] = _size_profile(holdout_labels)
                entry["out_of_sample_assignment"] = True
            else:
                entry["out_of_sample_assignment"] = False
                entry["out_of_sample_note"] = (
                    "Density-based fit has no out-of-sample assignment; train-partition "
                    "diagnostics only."
                )

            gate_failures: list[str] = []
            silhouette = entry["train_metrics"].get("silhouette")
            if profile["n_clusters"] < 2:
                gate_failures.append("fewer_than_two_clusters")
            if silhouette is None or silhouette < config.min_silhouette:
                gate_failures.append("silhouette_below_threshold")
            if profile["smallest_cluster_fraction"] < config.min_cluster_fraction:
                gate_failures.append("cluster_too_small")
            bootstrap_ari = entry["bootstrap_stability"].get("mean_ari")
            if bootstrap_ari is None or bootstrap_ari < config.min_bootstrap_ari:
                gate_failures.append("bootstrap_stability_below_threshold")
            seed_ari = entry["seed_stability"].get("mean_ari")
            if seed_ari is None or seed_ari < config.min_seed_ari:
                gate_failures.append("seed_stability_below_threshold")
            if entry["negative_controls"]["is_data_artefact"]:
                gate_failures.append(
                    "negative_control_dominated:"
                    + ",".join(entry["negative_controls"]["dominated_by"])
                )
            dissolved = entry["bootstrap_stability"].get("dissolved_clusters") or []
            if dissolved:
                gate_failures.append("clusters_dissolve_under_resampling:" + ",".join(dissolved))
            null_silhouette = entry["permutation_null"].get("mean_null_silhouette")
            if (
                silhouette is not None
                and null_silhouette is not None
                and silhouette - null_silhouette < config.min_silhouette_gain_over_null
            ):
                gate_failures.append("silhouette_not_better_than_permuted_null")
            outlier_ari = entry["outlier_sensitivity"].get("ari_vs_full_fit")
            if outlier_ari is not None and outlier_ari < config.min_outlier_sensitivity_ari:
                gate_failures.append("solution_driven_by_top_outliers")
            entry["gate_failures"] = gate_failures
            entry["passes_gates"] = not gate_failures
            entry["composite_score"] = (
                round(float((silhouette or 0.0) + (bootstrap_ari or 0.0)), 4)
                if not gate_failures
                else None
            )
            candidates.append(entry)

    passing = [item for item in candidates if item.get("passes_gates")]
    report: dict[str, Any] = {
        "report_type": "metaboguard_phenotype_clustering",
        "code_version": CODE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_type": "exploratory_unsupervised_phenotypes",
        "explanation_class": "model_association",
        "is_disease_classification": False,
        "labels_used_in_fit_or_selection": False,
        "dataset_fingerprint": di.file_fingerprint(Path(dataset_path)),
        "artifact": {
            "dir": str(artifact_dir),
            "model_name": inputs.artifact_metadata.get("model_name"),
            "run_label": inputs.artifact_metadata.get("run_label"),
            "code_version": inputs.artifact_metadata.get("code_version"),
            "frozen": True,
        },
        "space": config.space,
        "space_dimension": int(matrix_all.shape[1]),
        "split_source": inputs.split_source,
        "split_sizes": {name: int(len(idx)) for name, idx in inputs.splits.items()},
        "fit_rows": int(len(train_index)),
        "config": config.as_dict(),
        "candidates": candidates,
        "method_availability": {
            "kmeans": True,
            "gaussian_mixture": True,
            "hdbscan": HDBSCAN_AVAILABLE,
            "dbscan": True,
            "density_arm": "hdbscan" if HDBSCAN_AVAILABLE else "dbscan_fallback",
            "note": (
                "HDBSCAN is used from scikit-learn's own tree when available; otherwise the "
                "density arm degrades to DBSCAN. No extra dependency is installed either way."
            ),
        },
        "method_references": {
            "silhouette": "https://wis.kuleuven.be/stat/robust/papers/publications-1987/rousseeuw-silhouettes-jcam-sciencedirectopenarchiv.pdf",
            "hdbscan": "https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14",
            "consensus_clustering": "https://link.springer.com/article/10.1023/A:1023949509487",
            "bootstrap_clusterwise_jaccard": "https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf",
            "scikit_learn": "https://github.com/scikit-learn/scikit-learn",
        },
        "validity_caveats": [
            "A high silhouette can be produced by outliers rather than by real separation, which is why the top-1% outlier-sensitivity refit is a gate.",
            "Stability does not establish validity: a stable clustering can still be a stable artefact, which is why negative controls and a permuted null are also gates.",
        ],
        "warnings": [
            "Clusters are patient/metabolic phenotypes, not cancer diagnoses, subtypes or sites.",
            "Cross-sectional data: no future-risk interpretation is available.",
            "Survey weights are not applied, so clusters describe the analytic sample only.",
        ],
    }

    if not passing:
        report["status"] = "no_stable_clusters"
        report["selected"] = None
        report["abstain_reason"] = (
            "No candidate passed the quality, stability and negative-control gates. "
            "Reporting an unstable clustering as a phenotype finding would be misleading."
        )
        report["gate_failure_summary"] = {
            f"{item['method']}_k{item['k']}": item.get("gate_failures")
            for item in candidates
            if item.get("status") == "evaluated"
        }
    else:
        best = max(passing, key=lambda item: item["composite_score"])
        model = _fit(best["method"], best["k"], train_matrix, config, config.random_seed)
        train_labels = _labels(model, train_matrix)
        report["status"] = "stable_clusters_found"
        report["selected"] = {
            "method": best["method"],
            "k": best["k"],
            "composite_score": best["composite_score"],
            "selection_rule": (
                "Highest silhouette + bootstrap ARI among candidates passing every gate. "
                "No disease label participates in this choice."
            ),
            "train_metrics": best["train_metrics"],
            "holdout_metrics": best.get("holdout_metrics"),
            "bootstrap_stability": best["bootstrap_stability"],
            "seed_stability": best["seed_stability"],
            "negative_controls": best["negative_controls"],
            "size_profile": best["size_profile"],
        }
        report["characterisation"] = characterise(
            inputs, train_labels, model, train_matrix, train_index, config
        )
        if best.get("out_of_sample_assignment"):
            holdout_labels = _labels(model, holdout_matrix)
            report["holdout_characterisation"] = characterise(
                inputs, holdout_labels, model, holdout_matrix, holdout_index, config
            )
        report["projection"] = _projection(
            matrix_all[train_index], train_labels, config
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "clustering_report.json").write_text(json.dumps(report, indent=2))
    _write_chart_data(report, output)
    return report


def _projection(
    matrix: np.ndarray, labels: np.ndarray, config: ClusterConfig
) -> dict[str, Any]:
    """Seeded 2-D PCA projection of a subsample, for charting only (no identifiers)."""
    rng = np.random.default_rng(config.random_seed + 11)
    size = min(config.projection_sample, len(matrix))
    index = np.sort(rng.choice(len(matrix), size, replace=False))
    projector = PCA(n_components=2, random_state=config.random_seed)
    coordinates = projector.fit_transform(matrix[index])
    return {
        "method": "pca_2d_of_clustering_space",
        "explained_variance_ratio": [
            round(float(value), 4) for value in projector.explained_variance_ratio_
        ],
        "sample_size": int(size),
        "points": [
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "cluster": int(label),
            }
            for (x, y), label in zip(coordinates, labels[index])
        ],
        "note": "Aggregated visualisation sample. No participant identifiers are included.",
    }


def _write_chart_data(report: dict[str, Any], output: Path) -> None:
    """Persist the exact numbers behind each chart as CSV/JSON."""
    rows = [
        {
            "method": item["method"],
            "k": item["k"],
            "silhouette": item.get("train_metrics", {}).get("silhouette"),
            "davies_bouldin": item.get("train_metrics", {}).get("davies_bouldin"),
            "calinski_harabasz": item.get("train_metrics", {}).get("calinski_harabasz"),
            "bootstrap_mean_ari": item.get("bootstrap_stability", {}).get("mean_ari"),
            "seed_mean_ari": item.get("seed_stability", {}).get("mean_ari"),
            "smallest_cluster_fraction": item.get("size_profile", {}).get(
                "smallest_cluster_fraction"
            ),
            "n_clusters": item.get("size_profile", {}).get("n_clusters"),
            "passes_gates": item.get("passes_gates"),
            "gate_failures": ";".join(item.get("gate_failures", []) or []),
        }
        for item in report["candidates"]
        if item.get("status") == "evaluated"
    ]
    pd.DataFrame(rows).to_csv(output / "candidate_metrics.csv", index=False)

    controls = [
        {
            "method": item["method"],
            "k": item["k"],
            **{
                f"control_{name}": payload["value"]
                for name, payload in item.get("negative_controls", {})
                .get("controls", {})
                .items()
            },
            "dominated_by": ";".join(item.get("negative_controls", {}).get("dominated_by", [])),
        }
        for item in report["candidates"]
        if item.get("status") == "evaluated"
    ]
    pd.DataFrame(controls).to_csv(output / "negative_controls.csv", index=False)

    if report.get("projection"):
        pd.DataFrame(report["projection"]["points"]).to_csv(
            output / "projection_points.csv", index=False
        )
    if report.get("characterisation"):
        panels = [
            {
                "cluster_id": cluster["cluster_id"],
                "feature": item["feature"],
                "robust_standardised_difference": item["robust_standardised_difference"],
                "direction": item["direction"],
            }
            for cluster in report["characterisation"]["clusters"]
            for item in cluster["top_distinguishing_panel"]
        ]
        pd.DataFrame(panels).to_csv(output / "cluster_feature_panels.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"))
    parser.add_argument(
        "--artifact",
        default=str(
            PROJECT_ROOT
            / "model_artifacts"
            / "metaboguard_ssl"
            / "meeting_2026-08-04"
            / "ssl_artifact"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "model_artifacts" / "clustering" / "latest")
    )
    parser.add_argument("--space", choices=["latent", "features"], default="latent")
    parser.add_argument("--methods", default="kmeans,gaussian_mixture,hdbscan")
    parser.add_argument("--k-values", default="2,3,4,5,6,7,8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-rounds", type=int, default=20)
    parser.add_argument("--max-fit-rows", type=int, default=20_000)
    parser.add_argument(
        "--complete-cases-only",
        action="store_true",
        help="Sensitivity analysis: keep only rows with all usable_now features observed.",
    )
    arguments = parser.parse_args()

    config = ClusterConfig(
        space=arguments.space,
        methods=tuple(part.strip() for part in arguments.methods.split(",") if part.strip()),
        k_values=tuple(int(part) for part in arguments.k_values.split(",") if part.strip()),
        random_seed=arguments.seed,
        bootstrap_rounds=arguments.bootstrap_rounds,
        max_fit_rows=arguments.max_fit_rows,
        restrict_to_complete_cases=arguments.complete_cases_only,
    )
    report = run_clustering(
        arguments.dataset, arguments.artifact, arguments.output_dir, config
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "space": report["space"],
                "split_source": report["split_source"],
                "selected": (report.get("selected") or {}).get("method"),
                "k": (report.get("selected") or {}).get("k"),
                "silhouette": (report.get("selected") or {}).get("train_metrics", {}).get(
                    "silhouette"
                ),
                "bootstrap_mean_ari": (report.get("selected") or {})
                .get("bootstrap_stability", {})
                .get("mean_ari"),
                "negative_controls": (report.get("selected") or {}).get(
                    "negative_controls", {}
                ).get("controls"),
                "abstain_reason": report.get("abstain_reason"),
                "output_dir": arguments.output_dir,
                "reminder": "Exploratory phenotypes only - never a cancer diagnosis or site.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()