"""MetaboGuard self-supervised tabular representation and anomaly scoring.

The encoder is trained without disease labels using masked-feature
reconstruction. Disease labels are used only for post-hoc association checks.
Current NHANES data are cross-sectional, so outputs are metabolic deviation
scores, not future disease probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


PREVENTION_FEATURES = [
    "DEMO_RIDAGEYR",
    "DEMO_RIAGENDR",
    "DEMO_RIDRETH3",
    "BMX_BMXBMI",
    "BMX_BMXWAIST",
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "INS_LBXIN",
    "CPEP_LBXCPSI",
    "TRIGLY_LBXTR",
    "TRIGLY_LBDLDL",
    "HDL_LBDHDD",
    "TCHOL_LBXTC",
    "HSCRP_LBXHSCRP",
    "CBC_LBXHGB",
    "CBC_LBXPLTSI",
    "BIOPRO_LBXSATSI",
    "BIOPRO_LBXSAPSI",
    "BIOPRO_LBXSCR",
    "smoking_status",
    "alcohol_status",
    "average_drinks_per_day",
    "weight_loss_1yr_lb",
    "weight_loss_10yr_lb",
    "homa_ir",
]

CATEGORICAL_FEATURES = [
    "DEMO_RIAGENDR",
    "DEMO_RIDRETH3",
    "smoking_status",
    "alcohol_status",
]

FORBIDDEN_EARLY_WARNING_FEATURES = {
    "Cancer",
    "PancreaticCancer",
    "NODM_PancreaticCancer",
    "tcga_stage_ordinal",
    "tcga_grade_ordinal",
    "tcga_tumor_status",
    "tcga_treatment_response",
    "tcga_followup_days",
    "tcga_event",
    "tcga_pfi_days",
    "tcga_pfi_event",
}


@dataclass
class SSLConfig:
    latent_dim: int = 16
    hidden_dim: int = 96
    epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 1e-3
    mask_probability: float = 0.15
    gaussian_noise: float = 0.03
    validation_fraction: float = 0.1
    patience: int = 6
    max_train_rows: int = 50_000
    random_seed: int = 42


def dataset_capabilities(frame: pd.DataFrame) -> dict[str, Any]:
    repeated_id = None
    for candidate in ("patient_id", "person_id", "subject_id"):
        if candidate in frame.columns:
            repeated_id = candidate
            break
    has_repeated_patients = bool(
        repeated_id and frame[repeated_id].duplicated(keep=False).any()
    )
    time_columns = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("event_date", "diagnosis_date", "followup_days"))
    ]
    has_longitudinal_outcomes = has_repeated_patients and bool(time_columns)
    return {
        "rows": int(len(frame)),
        "repeated_patient_id": repeated_id,
        "has_repeated_patient_measurements": has_repeated_patients,
        "time_columns": time_columns,
        "supports_future_development_prediction": has_longitudinal_outcomes,
        "supported_output": (
            "multi_horizon_risk"
            if has_longitudinal_outcomes
            else "cross_sectional_representation_and_deviation_only"
        ),
        "warning": (
            None
            if has_longitudinal_outcomes
            else "No patient-level longitudinal outcomes: do not describe scores as future disease risk."
        ),
    }


def choose_supported_horizons(
    event_time_days: pd.Series,
    event: pd.Series,
    candidates: tuple[int, ...] = (365, 1095, 1825),
    minimum_events: int = 50,
    minimum_non_events: int = 50,
) -> list[int]:
    time = pd.to_numeric(event_time_days, errors="coerce")
    status = pd.to_numeric(event, errors="coerce")
    horizons: list[int] = []
    for horizon in candidates:
        events = int(((status == 1) & (time <= horizon)).sum())
        non_events = int((time >= horizon).sum())
        if events >= minimum_events and non_events >= minimum_non_events:
            horizons.append(horizon)
    return horizons


def select_prevention_features(frame: pd.DataFrame) -> list[str]:
    features = [
        feature for feature in PREVENTION_FEATURES
        if feature in frame.columns
        and feature not in FORBIDDEN_EARLY_WARNING_FEATURES
        and frame[feature].notna().any()
    ]
    if not features:
        raise ValueError("No prevention-safe features are available.")
    return features


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature not in categorical]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler(quantile_range=(10, 90))),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _torch_components(input_dim: int, config: SSLConfig):
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for training. Install the optional deep-learning dependencies."
        ) from error

    class DenoisingAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Linear(input_dim, config.hidden_dim)
            self.enc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
            self.enc_out = nn.Linear(config.hidden_dim // 2, config.latent_dim)
            self.dec1 = nn.Linear(config.latent_dim, config.hidden_dim // 2)
            self.dec2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim)
            self.dec_out = nn.Linear(config.hidden_dim, input_dim)
            self.activation = nn.GELU()

        def encode(self, values):
            values = self.activation(self.enc1(values))
            values = self.activation(self.enc2(values))
            return self.enc_out(values)

        def decode(self, latent):
            latent = self.activation(self.dec1(latent))
            latent = self.activation(self.dec2(latent))
            return self.dec_out(latent)

        def forward(self, values):
            latent = self.encode(values)
            return self.decode(latent), latent

    return torch, DenoisingAutoencoder


def _export_numpy_weights(model, output_path: Path) -> None:
    state = model.state_dict()
    arrays = {
        key.replace(".", "__"): value.detach().cpu().numpy()
        for key, value in state.items()
    }
    np.savez_compressed(output_path, **arrays)


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3))
    )


class NumpyAutoencoder:
    def __init__(self, weights_path: str | Path) -> None:
        archive = np.load(weights_path)
        self.weights = {key.replace("__", "."): archive[key] for key in archive.files}

    def _linear(self, values: np.ndarray, prefix: str) -> np.ndarray:
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights[f"{prefix}.bias"]
        return values @ weight.T + bias

    def encode(self, values: np.ndarray) -> np.ndarray:
        values = _gelu(self._linear(values, "enc1"))
        values = _gelu(self._linear(values, "enc2"))
        return self._linear(values, "enc_out")

    def reconstruct(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent = self.encode(values)
        decoded = _gelu(self._linear(latent, "dec1"))
        decoded = _gelu(self._linear(decoded, "dec2"))
        return self._linear(decoded, "dec_out"), latent


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    location = float(np.median(values))
    scale = float(np.median(np.abs(values - location)) * 1.4826)
    return location, max(scale, 1e-8)


def _transformed_source_map(
    transformed_names: list[str],
    raw_features: list[str],
) -> list[str]:
    mapping: list[str] = []
    for transformed in transformed_names:
        suffix = transformed.split("__", 1)[-1]
        suffix = suffix.removeprefix("missingindicator_")
        matches = [feature for feature in raw_features if suffix == feature or suffix.startswith(f"{feature}_")]
        mapping.append(max(matches, key=len) if matches else suffix)
    return mapping


def train_self_supervised(
    frame: pd.DataFrame,
    output_dir: str | Path,
    config: SSLConfig | None = None,
) -> dict[str, Any]:
    config = config or SSLConfig()
    rng = np.random.default_rng(config.random_seed)
    features = select_prevention_features(frame)
    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18].copy()
    if len(adult) < 500:
        raise ValueError("At least 500 adult records are required for self-supervised training.")

    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(adult[features]).astype(np.float32)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    transformed_sources = _transformed_source_map(transformed_names, features)
    finite = np.isfinite(transformed).all(axis=1)
    transformed = transformed[finite]
    adult = adult.loc[finite].copy()

    if len(transformed) > config.max_train_rows:
        selected = rng.choice(len(transformed), config.max_train_rows, replace=False)
        training_values = transformed[selected]
    else:
        training_values = transformed

    torch, Model = _torch_components(training_values.shape[1], config)
    torch.manual_seed(config.random_seed)
    model = Model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    loss_function = torch.nn.MSELoss()

    order = rng.permutation(len(training_values))
    validation_rows = max(1, int(len(order) * config.validation_fraction))
    validation = torch.tensor(training_values[order[:validation_rows]])
    training = torch.tensor(training_values[order[validation_rows:]])

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        permutation = torch.randperm(len(training))
        for start in range(0, len(training), config.batch_size):
            batch = training[permutation[start:start + config.batch_size]]
            mask = torch.rand_like(batch) < config.mask_probability
            corrupted = batch.clone()
            corrupted[mask] = 0.0
            corrupted = corrupted + torch.randn_like(corrupted) * config.gaussian_noise
            reconstructed, _ = model(corrupted)
            loss = loss_function(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            validation_reconstruction, _ = model(validation)
            validation_loss = float(loss_function(validation_reconstruction, validation))
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    _export_numpy_weights(model, output / "autoencoder_weights.npz")

    scorer = NumpyAutoencoder(output / "autoencoder_weights.npz")
    reconstructed, latent = scorer.reconstruct(transformed)
    reconstruction_error = np.mean((reconstructed - transformed) ** 2, axis=1)
    latent_mean = latent.mean(axis=0)
    latent_std = np.maximum(latent.std(axis=0), 1e-6)
    latent_distance = np.mean(((latent - latent_mean) / latent_std) ** 2, axis=1)
    rec_location, rec_scale = _robust_location_scale(reconstruction_error)
    latent_location, latent_scale = _robust_location_scale(latent_distance)
    combined = (
        0.7 * np.maximum(0, (reconstruction_error - rec_location) / rec_scale)
        + 0.3 * np.maximum(0, (latent_distance - latent_location) / latent_scale)
    )
    sorted_scores = np.sort(combined)

    capabilities = dataset_capabilities(frame)
    posthoc = evaluate_posthoc_associations(adult, latent)
    metadata = {
        "model_name": "metaboguard_ssl_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "intended_use": "Research-only metabolic deviation and representation learning.",
        "not_intended_for": [
            "diagnosis",
            "treatment decisions",
            "patient reassurance",
            "future disease probability without longitudinal validation",
        ],
        "features": features,
        "transformed_feature_names": transformed_names,
        "transformed_feature_sources": transformed_sources,
        "transformed_dimension": int(transformed.shape[1]),
        "training_rows": int(len(training_values)),
        "adult_rows_scored": int(len(adult)),
        "config": asdict(config),
        "training_history": history,
        "capabilities": capabilities,
        "score_distribution": {
            "reconstruction_location": rec_location,
            "reconstruction_scale": rec_scale,
            "latent_location": latent_location,
            "latent_scale": latent_scale,
            "latent_mean": latent_mean.tolist(),
            "latent_std": latent_std.tolist(),
            "combined_sorted": sorted_scores.tolist(),
            "percentile_thresholds": {
                "90": float(np.quantile(combined, 0.90)),
                "95": float(np.quantile(combined, 0.95)),
                "99": float(np.quantile(combined, 0.99)),
            },
        },
        "posthoc_association_checks": posthoc,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _head_target(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "any_cancer_prevalence":
        return pd.to_numeric(frame.get("Cancer"), errors="coerce")
    if name == "type2_diabetes_proxy":
        diabetes = pd.to_numeric(frame.get("Diabetes"), errors="coerce")
        subtype = pd.to_numeric(frame.get("diabetes_subtype"), errors="coerce")
        return pd.Series(
            np.where((diabetes == 1) & (subtype == 2), 1, np.where(diabetes == 0, 0, np.nan)),
            index=frame.index,
        )
    if name == "type1_diabetes_proxy_research_only":
        subtype = pd.to_numeric(frame.get("diabetes_subtype"), errors="coerce")
        return pd.Series(
            np.where(subtype == 1, 1, np.where(subtype == 0, 0, np.nan)),
            index=frame.index,
        )
    raise ValueError(name)


def evaluate_posthoc_associations(
    frame: pd.DataFrame,
    latent: np.ndarray,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in (
        "any_cancer_prevalence",
        "type2_diabetes_proxy",
        "type1_diabetes_proxy_research_only",
    ):
        target = _head_target(frame, name)
        valid = target.notna().to_numpy()
        y = target.loc[target.notna()].astype(int).to_numpy()
        x = latent[valid]
        positives = int(y.sum())
        negatives = int((y == 0).sum())
        if positives < 20 or negatives < 20:
            results[name] = {
                "status": "not_evaluated",
                "positives": positives,
                "negatives": negatives,
                "reason": "Fewer than 20 cases in one class.",
            }
            continue
        folds = min(5, positives)
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        )
        probability = cross_val_predict(
            classifier,
            x,
            y,
            cv=StratifiedKFold(folds, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        results[name] = {
            "status": "cross_sectional_association_only",
            "positives": positives,
            "negatives": negatives,
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "warning": "This does not measure future disease development.",
        }
        if name == "type1_diabetes_proxy_research_only":
            results[name]["warning"] = (
                "Research-only proxy defined from diagnosis age and insulin use; "
                "not a validated Type 1 endpoint or future-development measure."
            )
    return results


def score_records(
    records: pd.DataFrame,
    artifact_dir: str | Path,
) -> list[dict[str, Any]]:
    artifact = Path(artifact_dir)
    metadata = json.loads((artifact / "metadata.json").read_text())
    preprocessor = joblib.load(artifact / "preprocessor.joblib")
    features = metadata["features"]
    frame = records.copy()
    for feature in features:
        if feature not in frame:
            frame[feature] = np.nan
    transformed = preprocessor.transform(frame[features]).astype(np.float32)
    scorer = NumpyAutoencoder(artifact / "autoencoder_weights.npz")
    reconstructed, latent = scorer.reconstruct(transformed)
    reconstruction = np.mean((reconstructed - transformed) ** 2, axis=1)
    distribution = metadata["score_distribution"]
    latent_mean = np.asarray(distribution["latent_mean"])
    latent_std = np.asarray(distribution["latent_std"])
    latent_distance = np.mean(((latent - latent_mean) / latent_std) ** 2, axis=1)
    combined = (
        0.7 * np.maximum(
            0,
            (reconstruction - distribution["reconstruction_location"])
            / distribution["reconstruction_scale"],
        )
        + 0.3 * np.maximum(
            0,
            (latent_distance - distribution["latent_location"])
            / distribution["latent_scale"],
        )
    )
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
            "interpretation": (
                "Higher means more unusual relative to the training reference. "
                "It is not a cancer or diabetes diagnosis/probability."
            ),
        }
        for score, percentile, row, contributions in zip(
            combined, percentiles, latent, contribution_rows
        )
    ]
