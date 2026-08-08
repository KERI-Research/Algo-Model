#!/usr/bin/env python3
"""Export aggregate NHANES statistics for realistic browser-side profiles.

The output contains only empirical quantiles, category frequencies and a
regularized rank-correlation factor from the SSL training partition. It never
contains source rows, participant identifiers or local filesystem paths.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from data_integrity import file_fingerprint, group_split_indices
from data_reliability import PLAUSIBLE_RANGES
from self_supervised import CATEGORICAL_FEATURES, PREVENTION_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "frontend" / "src" / "interface" / "synthetic_profile_model.json"
SCHEMA_VERSION = "metaboguard-synthetic-profile-v1"
QUANTILE_GRID = np.linspace(0.01, 0.99, 99)
MIN_PROFILE_ROWS = 250
DERIVED_FIELDS = {"homa_ir", "TCHOL_LBXTC"}

PROFILE_DEFINITIONS = {
    "reference_range": {
        "label": "Reference-pattern adult",
        "description": (
            "A fabricated adult profile sampled from aggregate records without reported "
            "diabetes and without the selected metabolic threshold flags."
        ),
        "reported_diabetes": "0",
    },
    "metabolic_deviation": {
        "label": "Metabolically atypical adult",
        "description": (
            "A fabricated adult profile sampled from aggregate records without reported "
            "diabetes but with one or more selected metabolic threshold flags."
        ),
        "reported_diabetes": "0",
    },
    "reported_diabetes_metabolic": {
        "label": "Adult with reported diabetes",
        "description": (
            "A fabricated adult profile sampled from aggregate records with reported "
            "diabetes; the status is context, not a model prediction."
        ),
        "reported_diabetes": "1",
    },
}


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce")


def _profile_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    diabetes = _numeric(frame, "Diabetes").eq(1)
    bmi = _numeric(frame, "BMX_BMXBMI")
    hba1c = _numeric(frame, "GHB_LBXGH")
    glucose = _numeric(frame, "GLU_LBXGLU")
    triglycerides = _numeric(frame, "TRIGLY_LBXTR")
    hdl = _numeric(frame, "HDL_LBDHDD")

    threshold_flag = (
        bmi.ge(27.5)
        | hba1c.ge(5.7)
        | glucose.ge(100)
        | triglycerides.ge(150)
        | hdl.le(40)
    )
    observed_threshold = pd.concat(
        [bmi, hba1c, glucose, triglycerides, hdl], axis=1
    ).notna().any(axis=1)
    return {
        "reference_range": (~diabetes) & observed_threshold & (~threshold_flag),
        "metabolic_deviation": (~diabetes) & observed_threshold & threshold_flag,
        "reported_diabetes_metabolic": diabetes,
    }


def _bounded_values(frame: pd.DataFrame, field: str) -> np.ndarray:
    values = _numeric(frame, field).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bounds = PLAUSIBLE_RANGES.get(field)
    if bounds is not None:
        low, high = map(float, bounds)
        values = values[(values >= low) & (values <= high)]
    return values


def _quantiles(profile: pd.DataFrame, fallback: pd.DataFrame, field: str) -> list[float]:
    values = _bounded_values(profile, field)
    if values.size < 50:
        values = _bounded_values(fallback, field)
    if values.size == 0:
        raise ValueError(f"No usable values for synthetic field '{field}'.")
    return [round(float(value), 6) for value in np.quantile(values, QUANTILE_GRID)]


def _normal_scores(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    normal = NormalDist()
    columns: list[np.ndarray] = []
    for field in fields:
        values = _numeric(frame, field)
        ranks = values.rank(method="average", pct=True)
        uniforms = ranks.clip(0.001, 0.999).fillna(0.5).to_numpy(dtype=float)
        columns.append(
            np.fromiter((normal.inv_cdf(value) for value in uniforms), dtype=float)
        )
    return np.column_stack(columns)


def _correlation_factor(profile: pd.DataFrame, fields: list[str]) -> list[list[float]]:
    scores = _normal_scores(profile, fields)
    correlation = np.corrcoef(scores, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    regularized = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(regularized), 1e-12))
    regularized = regularized / np.outer(scale, scale)
    regularized = (regularized + regularized.T) / 2.0
    np.fill_diagonal(regularized, 1.0)
    factor = np.linalg.cholesky(regularized + np.eye(len(fields)) * 1e-8)
    return [[round(float(value), 8) for value in row] for row in factor]


def _category_distribution(
    profile: pd.DataFrame, fallback: pd.DataFrame, field: str
) -> dict[str, list[Any]]:
    values = _numeric(profile, field).dropna()
    if len(values) < 50:
        values = _numeric(fallback, field).dropna()
    counts = values.astype(int).value_counts().sort_index()
    probabilities = counts / counts.sum()
    return {
        "values": [str(value) for value in counts.index],
        "probabilities": [round(float(value), 8) for value in probabilities],
    }


def build_model(dataset: Path, random_seed: int = 42) -> dict[str, Any]:
    frame = pd.read_csv(dataset, low_memory=False)
    adult = frame[_numeric(frame, "DEMO_RIDAGEYR").ge(18)].reset_index(drop=True)
    split = group_split_indices(adult, fractions=(0.7, 0.15, 0.15), seed=random_seed)
    training = adult.iloc[split["train"]].reset_index(drop=True)
    masks = _profile_masks(training)

    categorical_fields = [
        field
        for field in PREVENTION_FEATURES
        if field in CATEGORICAL_FEATURES and field in training
    ]
    continuous_fields = [
        field
        for field in PREVENTION_FEATURES
        if field not in categorical_fields
        and field not in DERIVED_FIELDS
        and field in training
        and _bounded_values(training, field).size >= 50
    ]

    profiles: dict[str, Any] = {}
    for profile_id, definition in PROFILE_DEFINITIONS.items():
        profile = training.loc[masks[profile_id]].copy()
        if len(profile) < MIN_PROFILE_ROWS:
            raise ValueError(
                f"Synthetic profile '{profile_id}' has only {len(profile)} training rows; "
                f"at least {MIN_PROFILE_ROWS} are required."
            )
        profiles[profile_id] = {
            **definition,
            "training_rows": int(len(profile)),
            "sampling_weight": round(float(len(profile) / len(training)), 8),
            "quantiles": {
                field: _quantiles(profile, training, field)
                for field in continuous_fields
            },
            "correlation_cholesky": _correlation_factor(profile, continuous_fields),
            "categories": {
                field: _category_distribution(profile, training, field)
                for field in categorical_fields
            },
            "missingness": {
                field: round(float(_numeric(profile, field).isna().mean()), 6)
                for field in continuous_fields
            },
            "context_quantiles": (
                {
                    "diabetes_duration_years": _quantiles(
                        profile, training, "diabetes_duration_years"
                    )
                }
                if profile_id == "reported_diabetes_metabolic"
                else {}
            ),
        }

    fingerprint = file_fingerprint(dataset)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "dataset": fingerprint["name"],
            "dataset_sha256": fingerprint["sha256"],
            "partition": "participant-grouped training split",
            "split_seed": random_seed,
            "training_rows": int(len(training)),
        },
        "privacy": {
            "contains_source_rows": False,
            "contains_identifiers": False,
            "aggregation": "1st-99th empirical quantiles, category frequencies and rank correlation",
        },
        "quantile_probabilities": [round(float(value), 2) for value in QUANTILE_GRID],
        "continuous_fields": continuous_fields,
        "categorical_fields": categorical_fields,
        "derived_fields": sorted(DERIVED_FIELDS),
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    payload = build_model(arguments.dataset.resolve(), arguments.seed)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                "output": output.name,
                "bytes": output.stat().st_size,
                "profiles": {
                    key: value["training_rows"]
                    for key, value in payload["profiles"].items()
                },
                "continuous_fields": len(payload["continuous_fields"]),
                "categorical_fields": len(payload["categorical_fields"]),
                "contains_source_rows": payload["privacy"]["contains_source_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()