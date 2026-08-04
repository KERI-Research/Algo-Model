"""Dependency-light NumPy inference path for serverless hosting.

Why this exists
---------------
``server/core/self_supervised.py`` (vendored, authoritative) scores records by
unpickling the fitted scikit-learn ``ColumnTransformer`` with joblib. That is
correct, but scikit-learn drags in SciPy, and pandas + NumPy + SciPy +
scikit-learn together exceed the practical unzipped size budget of a Vercel
Hobby Python function.

The fitted preprocessor is a plain, fully describable transformation:

* numeric block: per-column median imputation with ``add_indicator=True``
  (binary "was missing" columns for the columns that had missing values during
  fitting), then ``RobustScaler(quantile_range=(10, 90))`` applied to the
  imputed values and the indicators alike - ``(x - centre) / scale``;
* categorical block: most-frequent imputation, then one-hot encoding with
  ``handle_unknown="ignore"`` (an unseen level becomes an all-zero group).

``prepare_assets.py`` exports those fitted constants to
``assets/ssl_artifact/preprocessor_params.json``, and this module replays them
with NumPy only. Model weights, the score definition and the reference
distribution are untouched.

Parity is enforced by tests: :func:`score_records` here is compared to the
vendored scikit-learn implementation row by row (see
``tests/test_inference_parity.py``). When the exported parameters are absent,
:func:`score_records` transparently falls back to the vendored path, so a
scikit-learn install keeps working unchanged.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

PARAMS_FILENAME = "preprocessor_params.json"


# ---------------------------------------------------------------------------
# Fitted-parameter export (needs scikit-learn; run offline, never at runtime)
# ---------------------------------------------------------------------------


def export_preprocessor_params(artifact_dir: str | Path) -> dict[str, Any]:
    """Describe the fitted ColumnTransformer as plain JSON-serialisable data."""
    import joblib  # imported lazily: not a runtime dependency

    artifact = Path(artifact_dir)
    preprocessor = joblib.load(artifact / "preprocessor.joblib")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "feature_names_in": [str(name) for name in preprocessor.feature_names_in_],
        "blocks": [],
        "note": (
            "Fitted constants exported from preprocessor.joblib so inference needs "
            "only NumPy and pandas. Values are the scikit-learn attributes verbatim."
        ),
    }
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        steps = dict(transformer.steps)
        if name == "numeric":
            imputer = steps["imputer"]
            scaler = steps["scale"]
            payload["blocks"].append(
                {
                    "kind": "numeric",
                    "columns": [str(column) for column in columns],
                    "imputer_statistics": [float(value) for value in imputer.statistics_],
                    "add_indicator": bool(imputer.add_indicator),
                    "indicator_columns": (
                        [int(index) for index in imputer.indicator_.features_]
                        if imputer.add_indicator
                        else []
                    ),
                    "centre": [float(value) for value in np.atleast_1d(scaler.center_)],
                    "scale": [float(value) for value in np.atleast_1d(scaler.scale_)],
                }
            )
        elif name == "categorical":
            imputer = steps["imputer"]
            encoder = steps["onehot"]
            payload["blocks"].append(
                {
                    "kind": "categorical",
                    "columns": [str(column) for column in columns],
                    "imputer_statistics": [
                        _jsonable(value) for value in imputer.statistics_
                    ],
                    "categories": [
                        [_jsonable(level) for level in levels]
                        for levels in encoder.categories_
                    ],
                    "handle_unknown": encoder.handle_unknown,
                }
            )
        else:  # pragma: no cover - guards against an artifact change
            raise ValueError(f"Unsupported preprocessor block '{name}'.")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Runtime: NumPy-only transform
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _load_params(artifact_dir: str) -> dict[str, Any] | None:
    path = Path(artifact_dir) / PARAMS_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def params_available(artifact_dir: str | Path | None = None) -> bool:
    directory = str(artifact_dir or config.SSL_ARTIFACT_DIR)
    return _load_params(directory) is not None


def _numeric_block(frame: pd.DataFrame, block: dict[str, Any]) -> np.ndarray:
    columns = block["columns"]
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    matrix = values.to_numpy(dtype=np.float64, copy=True)
    statistics = np.asarray(block["imputer_statistics"], dtype=np.float64)
    missing = np.isnan(matrix)
    if missing.any():
        matrix[missing] = np.take(statistics, np.where(missing)[1])
    if block.get("add_indicator"):
        indicator_columns = block.get("indicator_columns", [])
        indicators = missing[:, indicator_columns].astype(np.float64)
        matrix = np.hstack([matrix, indicators])
    centre = np.asarray(block["centre"], dtype=np.float64)
    scale = np.asarray(block["scale"], dtype=np.float64)
    return (matrix - centre) / scale


def _categorical_block(frame: pd.DataFrame, block: dict[str, Any]) -> np.ndarray:
    columns = block["columns"]
    rows = len(frame)
    encoded: list[np.ndarray] = []
    for position, column in enumerate(columns):
        levels = block["categories"][position]
        fallback = block["imputer_statistics"][position]
        series = (
            frame[column]
            if column in frame.columns
            else pd.Series([np.nan] * rows, index=frame.index)
        )
        # SimpleImputer(missing_values=np.nan) only replaces float NaN / pandas NA.
        # In an object column, None is NOT missing to scikit-learn: it reaches the
        # encoder as an unseen category and, under handle_unknown="ignore",
        # produces an all-zero group. Mirror that exactly.
        keys = [
            _level_key(fallback) if _is_nan(value) else _level_key(value)
            for value in series.tolist()
        ]
        index = {_level_key(level): number for number, level in enumerate(levels)}
        group = np.zeros((rows, len(levels)), dtype=np.float64)
        for row, key in enumerate(keys):
            column_position = index.get(key)
            if column_position is not None:
                group[row, column_position] = 1.0
            # handle_unknown="ignore": an unseen level stays all-zero.
        encoded.append(group)
    if not encoded:
        return np.zeros((rows, 0), dtype=np.float64)
    return np.hstack(encoded)


def _is_nan(value: Any) -> bool:
    """True only for float NaN or pandas NA - deliberately not for None."""
    if value is pd.NA:
        return True
    return isinstance(value, float) and np.isnan(value)


def _level_key(value: Any) -> str:
    """Match scikit-learn's category identity for numeric-looking labels."""
    if value is None:
        return "\x00none"
    if isinstance(value, float) and np.isnan(value):
        return "\x00nan"
    if isinstance(value, (bool, np.bool_)):
        return f"b:{bool(value)}"
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric.is_integer():
            return f"n:{int(numeric)}"
        return f"n:{numeric!r}"
    # Strings are NOT coerced to numbers: scikit-learn compares categories as
    # Python objects, so "1" never matches the fitted level 1.0 and becomes an
    # ignored unknown level. Numeric ints and floats do compare equal (1 == 1.0),
    # which the "n:" normalisation above preserves.
    return f"s:{str(value).strip()}"


def transform(frame: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """Apply the exported preprocessor to a frame of raw feature columns."""
    blocks = [
        _numeric_block(frame, block) if block["kind"] == "numeric"
        else _categorical_block(frame, block)
        for block in params["blocks"]
    ]
    return np.hstack(blocks)


# ---------------------------------------------------------------------------
# Runtime: autoencoder and score assembly (mirrors the vendored implementation)
# ---------------------------------------------------------------------------


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3))
    )


@lru_cache(maxsize=2)
def _load_weights(weights_path: str) -> dict[str, np.ndarray]:
    archive = np.load(weights_path)
    return {key.replace("__", "."): archive[key] for key in archive.files}


def _linear(values: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return values @ weights[f"{prefix}.weight"].T + weights[f"{prefix}.bias"]


def _reconstruct(values: np.ndarray, weights: dict[str, np.ndarray]):
    latent = _gelu(_linear(values, weights, "enc1"))
    latent = _gelu(_linear(latent, weights, "enc2"))
    latent = _linear(latent, weights, "enc_out")
    decoded = _gelu(_linear(latent, weights, "dec1"))
    decoded = _gelu(_linear(decoded, weights, "dec2"))
    return _linear(decoded, weights, "dec_out"), latent


def _deviation_scores(
    reconstruction_error: np.ndarray,
    latent_distance: np.ndarray,
    reference: dict[str, float],
) -> np.ndarray:
    return 0.7 * np.maximum(
        0,
        (reconstruction_error - reference["reconstruction_location"])
        / reference["reconstruction_scale"],
    ) + 0.3 * np.maximum(
        0,
        (latent_distance - reference["latent_location"]) / reference["latent_scale"],
    )


def score_records(
    records: pd.DataFrame,
    artifact_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Score records with the exported NumPy path, or the vendored path if absent.

    Output is identical, field for field, to
    ``server.core.self_supervised.score_records``.
    """
    artifact = Path(artifact_dir or config.SSL_ARTIFACT_DIR)
    params = _load_params(str(artifact))
    if params is None:
        from .core_bridge import score_records as sklearn_score_records

        return sklearn_score_records(records, artifact)

    metadata = json.loads((artifact / "metadata.json").read_text())
    features = metadata["features"]
    frame = records.copy()
    for feature in features:
        if feature not in frame:
            frame[feature] = np.nan
    transformed = transform(frame[features], params).astype(np.float32)

    weights = _load_weights(str(artifact / "autoencoder_weights.npz"))
    reconstructed, latent = _reconstruct(transformed, weights)
    reconstruction = np.mean((reconstructed - transformed) ** 2, axis=1)

    distribution = metadata["score_distribution"]
    latent_mean = np.asarray(distribution["latent_mean"])
    latent_std = np.asarray(distribution["latent_std"])
    latent_distance = np.mean(((latent - latent_mean) / latent_std) ** 2, axis=1)
    combined = _deviation_scores(reconstruction, latent_distance, distribution)
    reference = np.asarray(distribution["combined_sorted"])
    percentiles = np.searchsorted(reference, combined, side="right") / len(reference)

    squared_error = (reconstructed - transformed) ** 2
    source_names = metadata["transformed_feature_sources"]
    contribution_rows: list[list[dict[str, Any]]] = []
    for row_error in squared_error:
        grouped: dict[str, list[float]] = {}
        for source, error in zip(source_names, row_error):
            grouped.setdefault(source, []).append(float(error))
        ranked = sorted(
            (
                {"feature": source, "reconstruction_error": float(np.mean(errors))}
                for source, errors in grouped.items()
            ),
            key=lambda item: item["reconstruction_error"],
            reverse=True,
        )
        contribution_rows.append(ranked[:5])

    return [
        {
            "metabolic_deviation_score": round(float(score), 6),
            "reference_percentile": round(float(percentile * 100), 2),
            "latent_representation": row.tolist(),
            "top_deviation_features": contributions,
            "output_type": "metabolic_deviation_and_representation",
            "is_future_risk_probability": False,
            "interpretation": (
                "Higher means more unusual relative to the training reference. "
                "It is not a cancer or diabetes diagnosis/probability."
            ),
        }
        for score, percentile, row, contributions in zip(
            combined, percentiles, latent, contribution_rows
        )
    ]
