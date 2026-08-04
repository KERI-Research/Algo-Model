"""MetaboGuard self-supervised tabular representation and deviation scoring.

The encoder is a denoising autoencoder trained **without any disease label**.
Disease labels are used only for clearly-flagged post-hoc, cross-sectional
association checks on a held-out partition. Current NHANES files are
cross-sectional, so the outputs are metabolic *deviation scores* and latent
representations - never future disease probabilities.

Design decisions (documented in docs/METHODOLOGY.md):

* **Participant-grouped, seeded splits.** ``data_integrity.group_split_indices``
  produces 70/15/15 train/validation/holdout partitions grouped by participant
  identifier, so the pipeline stays correct if longitudinal data arrive.
* **Preprocessing is fit on the training partition only.** Imputation medians and
  robust scaling never see validation/holdout rows.
* **Deviation reference statistics come from the training partition only**, so a
  percentile is interpretable as "unusual relative to the training reference".
* **Two interchangeable backends.** ``torch`` when PyTorch is installed,
  otherwise a deterministic NumPy Adam implementation of the same architecture.
  Both export identical weight names, so scoring code is backend-agnostic.
* **Fail-closed capability gates.** Future-risk heads cannot be enabled unless a
  horizon passes the 50-event / 50-non-event gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
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

CODE_VERSION = "metaboguard-ssl-v1.1"

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
    "Diabetes",
    "diabetes_subtype",
    "new_onset_diabetes",
    "pancreatic_cancer_diagnosis_age",
    "pancreatic_cancer_minus_diabetes_years",
    "same_year_diabetes_pancreatic_cancer",
    "tcga_stage_ordinal",
    "tcga_grade_ordinal",
    "tcga_tumor_status",
    "tcga_treatment_response",
    "tcga_followup_days",
    "tcga_event",
    "tcga_pfi_days",
    "tcga_pfi_event",
}

#: Minimum class counts before a post-hoc association check is reported at all.
MIN_ASSOCIATION_CASES = 50


@dataclass
class SSLConfig:
    latent_dim: int = 16
    hidden_dim: int = 96
    epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    mask_probability: float = 0.15
    gaussian_noise: float = 0.03
    patience: int = 6
    max_train_rows: int = 50_000
    random_seed: int = 42
    #: "auto" prefers torch and falls back to the NumPy backend.
    backend: str = "auto"
    #: "cpu" is the default for reproducibility. "mps" uses Apple Silicon GPU
    #: (torch backend only) and is not bit-for-bit reproducible.
    device: str = "cpu"
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    #: Write a resumable checkpoint every N epochs (0 disables checkpointing).
    checkpoint_every: int = 5
    resume: bool = False
    minimum_adult_rows: int = 500
    run_label: str = "full"

    @classmethod
    def from_file(cls, path: str | Path) -> "SSLConfig":
        payload = json.loads(Path(path).read_text())
        known = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        if "split_fractions" in known:
            known["split_fractions"] = tuple(known["split_fractions"])
        unknown = sorted(set(payload) - set(known))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        return cls(**known)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split_fractions"] = list(self.split_fractions)
        return payload


# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------


def dataset_capabilities(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe, honestly, what this dataset can and cannot support."""
    participant_id = None
    for candidate in (
        "global_participant_id",
        "patient_id",
        "person_id",
        "subject_id",
        "bcr_patient_barcode",
        "SEQN",
    ):
        if candidate in frame.columns:
            participant_id = candidate
            break
    has_repeated_patients = bool(
        participant_id and frame[participant_id].duplicated(keep=False).any()
    )
    time_columns = [
        column
        for column in frame.columns
        if any(
            token in column.lower()
            for token in ("event_date", "diagnosis_date", "followup_days", "event_time_days")
        )
    ]
    has_longitudinal_outcomes = has_repeated_patients and bool(time_columns)
    return {
        "rows": int(len(frame)),
        "participant_id_column": participant_id,
        # Retained for backward compatibility with existing API consumers.
        "repeated_patient_id": participant_id if has_repeated_patients else None,
        "has_repeated_patient_measurements": has_repeated_patients,
        "time_columns": time_columns,
        "supports_future_development_prediction": has_longitudinal_outcomes,
        "supported_output": (
            "multi_horizon_risk"
            if has_longitudinal_outcomes
            else "cross_sectional_representation_and_deviation_only"
        ),
        "longitudinal_heads_enabled": False,
        "output_vocabulary": {
            "metabolic_deviation_score": "How unusual a metabolic profile is versus the training reference.",
            "reference_percentile": "Rank of that deviation score within the training reference distribution.",
            "latent_representation": "16-dimensional learned encoding of the input features.",
            "cross_sectional_association": "Association with an already-present condition. Not future risk.",
        },
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
    """Return the horizons (in days) that pass the event-count safety gate."""
    time = pd.to_numeric(event_time_days, errors="coerce")
    status = pd.to_numeric(event, errors="coerce")
    horizons: list[int] = []
    for horizon in candidates:
        events = int(((status == 1) & (time <= horizon)).sum())
        non_events = int((time >= horizon).sum())
        if events >= minimum_events and non_events >= minimum_non_events:
            horizons.append(horizon)
    return horizons


def assert_longitudinal_capability(frame: pd.DataFrame) -> None:
    """Fail closed when a future-risk head is requested on unsupported data."""
    capabilities = dataset_capabilities(frame)
    if not capabilities["supports_future_development_prediction"]:
        raise ValueError(
            "Future-risk (longitudinal) heads are disabled: this dataset has no "
            "patient-level follow-up. Outputs are deviation/representation only."
        )


# ---------------------------------------------------------------------------
# Features and preprocessing
# ---------------------------------------------------------------------------


def select_prevention_features(frame: pd.DataFrame) -> list[str]:
    """Allowlist-then-denylist feature selection with a hard leakage assertion."""
    from data_integrity import is_denylisted_input

    features = [
        feature
        for feature in PREVENTION_FEATURES
        if feature in frame.columns
        and feature not in FORBIDDEN_EARLY_WARNING_FEATURES
        and not is_denylisted_input(feature)
        and frame[feature].notna().any()
    ]
    if not features:
        raise ValueError("No prevention-safe features are available.")
    leaked = [feature for feature in features if is_denylisted_input(feature)]
    if leaked:
        raise AssertionError(f"Denylisted columns reached the feature set: {leaked}")
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


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3))
    )


def _gelu_gradient(values: np.ndarray) -> np.ndarray:
    root = math.sqrt(2.0 / math.pi)
    inner = root * (values + 0.044715 * values ** 3)
    tanh_inner = np.tanh(inner)
    d_inner = root * (1.0 + 3 * 0.044715 * values ** 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * values * (1.0 - tanh_inner ** 2) * d_inner


LAYER_ORDER = ("enc1", "enc2", "enc_out", "dec1", "dec2", "dec_out")


def _layer_dimensions(input_dim: int, config: SSLConfig) -> list[tuple[str, int, int]]:
    hidden = config.hidden_dim
    half = max(2, hidden // 2)
    return [
        ("enc1", input_dim, hidden),
        ("enc2", hidden, half),
        ("enc_out", half, config.latent_dim),
        ("dec1", config.latent_dim, half),
        ("dec2", half, hidden),
        ("dec_out", hidden, input_dim),
    ]


class NumpyDenoisingAutoencoder:
    """Deterministic NumPy implementation of the torch architecture (Adam)."""

    def __init__(self, input_dim: int, config: SSLConfig, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.config = config
        self.dimensions = _layer_dimensions(input_dim, config)
        self.parameters: dict[str, np.ndarray] = {}
        for name, fan_in, fan_out in self.dimensions:
            bound = 1.0 / math.sqrt(fan_in)
            self.parameters[f"{name}.weight"] = rng.uniform(
                -bound, bound, size=(fan_out, fan_in)
            ).astype(np.float32)
            self.parameters[f"{name}.bias"] = rng.uniform(
                -bound, bound, size=(fan_out,)
            ).astype(np.float32)
        self._moment1 = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self._moment2 = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self._step = 0

    # -- forward / backward -------------------------------------------------
    def _linear(self, values: np.ndarray, name: str) -> np.ndarray:
        return values @ self.parameters[f"{name}.weight"].T + self.parameters[f"{name}.bias"]

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        cache: dict[str, np.ndarray] = {"input": values}
        current = values
        for name in ("enc1", "enc2"):
            pre = self._linear(current, name)
            cache[f"{name}.pre"] = pre
            cache[f"{name}.in"] = current
            current = _gelu(pre)
        cache["enc_out.in"] = current
        latent = self._linear(current, "enc_out")
        current = latent
        for name in ("dec1", "dec2"):
            pre = self._linear(current, name)
            cache[f"{name}.pre"] = pre
            cache[f"{name}.in"] = current
            current = _gelu(pre)
        cache["dec_out.in"] = current
        reconstruction = self._linear(current, "dec_out")
        return reconstruction, latent, cache

    def encode(self, values: np.ndarray) -> np.ndarray:
        return self.forward(values)[1]

    def training_step(self, corrupted: np.ndarray, target: np.ndarray) -> float:
        reconstruction, _, cache = self.forward(corrupted)
        rows = target.shape[0]
        size = target.size
        loss = float(np.mean((reconstruction - target) ** 2))
        gradient = (2.0 / size) * (reconstruction - target)
        gradients: dict[str, np.ndarray] = {}

        def linear_backward(name: str, upstream: np.ndarray) -> np.ndarray:
            inputs = cache[f"{name}.in"]
            gradients[f"{name}.weight"] = upstream.T @ inputs
            gradients[f"{name}.bias"] = upstream.sum(axis=0)
            return upstream @ self.parameters[f"{name}.weight"]

        gradient = linear_backward("dec_out", gradient)
        for name in ("dec2", "dec1"):
            gradient = gradient * _gelu_gradient(cache[f"{name}.pre"])
            gradient = linear_backward(name, gradient)
        gradient = linear_backward("enc_out", gradient)
        for name in ("enc2", "enc1"):
            gradient = gradient * _gelu_gradient(cache[f"{name}.pre"])
            gradient = linear_backward(name, gradient)
        self._adam(gradients)
        del rows
        return loss

    def _adam(self, gradients: dict[str, np.ndarray]) -> None:
        self._step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        learning_rate = self.config.learning_rate
        for key, gradient in gradients.items():
            if self.config.weight_decay and key.endswith(".weight"):
                gradient = gradient + self.config.weight_decay * self.parameters[key]
            self._moment1[key] = beta1 * self._moment1[key] + (1 - beta1) * gradient
            self._moment2[key] = beta2 * self._moment2[key] + (1 - beta2) * gradient ** 2
            corrected1 = self._moment1[key] / (1 - beta1 ** self._step)
            corrected2 = self._moment2[key] / (1 - beta2 ** self._step)
            self.parameters[key] = (
                self.parameters[key]
                - learning_rate * corrected1 / (np.sqrt(corrected2) + epsilon)
            ).astype(np.float32)

    # -- state --------------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.parameters.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.parameters = {key: np.asarray(value, dtype=np.float32) for key, value in state.items()}


def _torch_components(input_dim: int, config: SSLConfig):
    import torch
    from torch import nn

    dimensions = dict(
        (name, (fan_in, fan_out)) for name, fan_in, fan_out in _layer_dimensions(input_dim, config)
    )

    class DenoisingAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in LAYER_ORDER:
                fan_in, fan_out = dimensions[name]
                setattr(self, name, nn.Linear(fan_in, fan_out))
            self.activation = nn.GELU(approximate="tanh")

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


def resolve_backend(requested: str) -> str:
    """Pick a training backend, preferring torch when it is importable."""
    if requested not in {"auto", "torch", "numpy"}:
        raise ValueError(f"Unknown backend '{requested}'.")
    if requested == "numpy":
        return "numpy"
    try:
        import torch  # noqa: F401
    except Exception as error:  # pragma: no cover - depends on environment
        if requested == "torch":
            raise RuntimeError(
                "backend='torch' requested but PyTorch is not installed. "
                "Install requirements-ssl.txt or use backend='numpy'."
            ) from error
        return "numpy"
    return "torch"


def resolve_device(requested: str, backend: str) -> str:
    """CPU by default. Apple Silicon 'mps' is opt-in and not bit-reproducible."""
    if backend == "numpy":
        return "cpu"
    import torch

    if requested == "auto":
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _export_numpy_weights(state: dict[str, Any], output_path: Path) -> None:
    arrays = {
        key.replace(".", "__"): np.asarray(value, dtype=np.float32)
        for key, value in state.items()
    }
    np.savez_compressed(output_path, **arrays)


class NumpyAutoencoder:
    """Backend-independent scorer that reads exported weights."""

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
        matches = [
            feature
            for feature in raw_features
            if suffix == feature or suffix.startswith(f"{feature}_")
        ]
        mapping.append(max(matches, key=len) if matches else suffix)
    return mapping


def deviation_scores(
    reconstruction_error: np.ndarray,
    latent_distance: np.ndarray,
    reference: dict[str, float],
) -> np.ndarray:
    """Combine reconstruction and latent-distance deviation (0.7 / 0.3)."""
    return 0.7 * np.maximum(
        0,
        (reconstruction_error - reference["reconstruction_location"])
        / reference["reconstruction_scale"],
    ) + 0.3 * np.maximum(
        0,
        (latent_distance - reference["latent_location"]) / reference["latent_scale"],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _package_versions() -> dict[str, str]:
    """Record the versions that produced an artifact (reproducibility manifest)."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "not_installed"
    return versions


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return output / "checkpoint_weights.npz", output / "checkpoint_state.json"


def _save_checkpoint(output: Path, state: dict[str, Any], progress: dict[str, Any]) -> None:
    weights_path, state_path = _checkpoint_paths(output)
    _export_numpy_weights(state, weights_path)
    state_path.write_text(json.dumps(progress, indent=2))


def _load_checkpoint(output: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    weights_path, state_path = _checkpoint_paths(output)
    if not (weights_path.exists() and state_path.exists()):
        return None
    archive = np.load(weights_path)
    state = {key.replace("__", "."): archive[key] for key in archive.files}
    return state, json.loads(state_path.read_text())


def _train_numpy(
    training: np.ndarray,
    validation: np.ndarray,
    config: SSLConfig,
    output: Path,
    start_epoch: int,
    initial_state: dict[str, np.ndarray] | None,
    history: list[dict[str, float]],
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    model = NumpyDenoisingAutoencoder(training.shape[1], config, config.random_seed)
    if initial_state:
        model.load_state_dict(initial_state)
    rng = np.random.default_rng(config.random_seed + 1)
    best_loss = min((entry["validation_loss"] for entry in history), default=float("inf"))
    best_state = model.state_dict()
    stale_epochs = 0
    for epoch in range(start_epoch, config.epochs):
        permutation = rng.permutation(len(training))
        losses: list[float] = []
        for start in range(0, len(training), config.batch_size):
            batch = training[permutation[start:start + config.batch_size]]
            mask = rng.random(batch.shape) < config.mask_probability
            corrupted = np.where(mask, 0.0, batch).astype(np.float32)
            corrupted = corrupted + rng.normal(
                0.0, config.gaussian_noise, size=batch.shape
            ).astype(np.float32)
            losses.append(model.training_step(corrupted, batch))
        reconstruction, _, _ = model.forward(validation)
        validation_loss = float(np.mean((reconstruction - validation) ** 2))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = model.state_dict()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
            _save_checkpoint(
                output,
                model.state_dict(),
                {"next_epoch": epoch + 1, "history": history, "backend": "numpy"},
            )
        if stale_epochs >= config.patience:
            break
    return best_state, history


def _train_torch(
    training_values: np.ndarray,
    validation_values: np.ndarray,
    config: SSLConfig,
    output: Path,
    device_name: str,
    start_epoch: int,
    initial_state: dict[str, np.ndarray] | None,
    history: list[dict[str, float]],
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    torch, Model = _torch_components(training_values.shape[1], config)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = torch.device(device_name)
    model = Model().to(device)
    if initial_state:
        model.load_state_dict(
            {key: torch.tensor(value) for key, value in initial_state.items()}
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = torch.nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(config.random_seed)
    training = torch.tensor(training_values, device=device)
    validation = torch.tensor(validation_values, device=device)

    best_loss = min((entry["validation_loss"] for entry in history), default=float("inf"))
    best_state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    stale_epochs = 0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_losses: list[float] = []
        permutation = torch.randperm(len(training), generator=generator).to(device)
        for start in range(0, len(training), config.batch_size):
            batch = training[permutation[start:start + config.batch_size]]
            mask = torch.rand(batch.shape, generator=generator).to(device) < config.mask_probability
            corrupted = batch.clone()
            corrupted[mask] = 0.0
            corrupted = corrupted + torch.randn(
                batch.shape, generator=generator
            ).to(device) * config.gaussian_noise
            reconstructed, _ = model(corrupted)
            loss = loss_function(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            reconstruction, _ = model(validation)
            validation_loss = float(loss_function(reconstruction, validation))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_loss": validation_loss,
            }
        )
        current_state = {
            key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
        }
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = current_state
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
            _save_checkpoint(
                output,
                current_state,
                {"next_epoch": epoch + 1, "history": history, "backend": "torch"},
            )
        if stale_epochs >= config.patience:
            break
    return best_state, history


def train_self_supervised(
    frame: pd.DataFrame,
    output_dir: str | Path,
    config: SSLConfig | None = None,
    dataset_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the label-free encoder and write a versioned artifact directory."""
    from data_integrity import group_split_indices, identifier_column

    config = config or SSLConfig()
    started_at = time.perf_counter()
    features = select_prevention_features(frame)
    label_columns = sorted(set(frame.columns) & FORBIDDEN_EARLY_WARNING_FEATURES)
    assert not (set(features) & set(label_columns)), "Outcome labels reached the encoder inputs."

    adult = frame[pd.to_numeric(frame["DEMO_RIDAGEYR"], errors="coerce") >= 18].copy()
    adult = adult.reset_index(drop=True)
    if len(adult) < config.minimum_adult_rows:
        raise ValueError(
            f"At least {config.minimum_adult_rows} adult records are required "
            f"for self-supervised training (got {len(adult)})."
        )

    splits = group_split_indices(
        adult, fractions=tuple(config.split_fractions), seed=config.random_seed
    )
    preprocessor = build_preprocessor(features)
    # Fit on the training partition only: no validation/holdout statistics leak.
    preprocessor.fit(adult.iloc[splits["train"]][features])
    transformed_all = preprocessor.transform(adult[features]).astype(np.float32)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    transformed_sources = _transformed_source_map(transformed_names, features)

    finite = np.isfinite(transformed_all).all(axis=1)
    keep = np.flatnonzero(finite)
    position_of = {position: index for index, position in enumerate(keep)}
    transformed = transformed_all[keep]
    adult = adult.iloc[keep].reset_index(drop=True)
    splits = {
        name: np.array([position_of[p] for p in indices if p in position_of], dtype=int)
        for name, indices in splits.items()
    }

    rng = np.random.default_rng(config.random_seed)
    train_index = splits["train"]
    if len(train_index) > config.max_train_rows:
        train_index = np.sort(
            rng.choice(train_index, config.max_train_rows, replace=False)
        )
    training_values = transformed[train_index]
    validation_values = transformed[splits["validation"]]
    if len(validation_values) < 32:
        raise ValueError("Validation partition is too small for early stopping.")

    backend = resolve_backend(config.backend)
    device_name = resolve_device(config.device, backend)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    initial_state: dict[str, np.ndarray] | None = None
    history: list[dict[str, float]] = []
    if config.resume:
        checkpoint = _load_checkpoint(output)
        if checkpoint is not None:
            initial_state, progress = checkpoint
            if progress.get("backend") != backend:
                raise ValueError(
                    f"Checkpoint was produced by backend '{progress.get('backend')}' "
                    f"but '{backend}' is active. Delete the checkpoint or switch backends."
                )
            start_epoch = int(progress.get("next_epoch", 0))
            history = list(progress.get("history", []))

    if backend == "numpy":
        best_state, history = _train_numpy(
            training_values, validation_values, config, output, start_epoch, initial_state, history
        )
    else:
        best_state, history = _train_torch(
            training_values,
            validation_values,
            config,
            output,
            device_name,
            start_epoch,
            initial_state,
            history,
        )

    training_seconds = time.perf_counter() - started_at
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    _export_numpy_weights(best_state, output / "autoencoder_weights.npz")
    # Persist the exact row partitions so any later evaluation reuses identical
    # split boundaries (positions refer to the adult, finite-row frame order).
    np.savez_compressed(
        output / "splits.npz",
        train=splits["train"],
        validation=splits["validation"],
        holdout=splits["holdout"],
        train_used_for_fit=train_index,
    )

    scorer = NumpyAutoencoder(output / "autoencoder_weights.npz")
    reconstructed, latent = scorer.reconstruct(transformed)
    reconstruction_error = np.mean((reconstructed - transformed) ** 2, axis=1)
    # Reference statistics are computed on the training partition only.
    latent_mean = latent[train_index].mean(axis=0)
    latent_std = np.maximum(latent[train_index].std(axis=0), 1e-6)
    latent_distance = np.mean(((latent - latent_mean) / latent_std) ** 2, axis=1)
    rec_location, rec_scale = _robust_location_scale(reconstruction_error[train_index])
    latent_location, latent_scale = _robust_location_scale(latent_distance[train_index])
    reference = {
        "reconstruction_location": rec_location,
        "reconstruction_scale": rec_scale,
        "latent_location": latent_location,
        "latent_scale": latent_scale,
    }
    combined = deviation_scores(reconstruction_error, latent_distance, reference)
    training_reference = np.sort(combined[train_index])

    capabilities = dataset_capabilities(frame)
    posthoc = evaluate_posthoc_associations(
        adult.iloc[splits["holdout"]], latent[splits["holdout"]]
    )
    metadata: dict[str, Any] = {
        "model_name": "metaboguard_ssl_v1",
        "code_version": CODE_VERSION,
        "artifact_schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "output_type": "metabolic_deviation_and_representation",
        "intended_use": "Research-only metabolic deviation and representation learning.",
        "not_intended_for": [
            "diagnosis",
            "treatment decisions",
            "patient reassurance",
            "future disease probability without longitudinal validation",
        ],
        "run_label": config.run_label,
        "backend": backend,
        "device": device_name,
        "dataset_fingerprint": dataset_fingerprint,
        "features": features,
        "label_columns_present_but_unused_in_training": label_columns,
        "transformed_feature_names": transformed_names,
        "transformed_feature_sources": transformed_sources,
        "transformed_dimension": int(transformed.shape[1]),
        "latent_dim": config.latent_dim,
        "training_rows": int(len(training_values)),
        "validation_rows": int(len(validation_values)),
        "holdout_rows": int(len(splits["holdout"])),
        "adult_rows_scored": int(len(adult)),
        "rows_dropped_non_finite": int(len(transformed_all) - len(transformed)),
        "split_policy": {
            "grouped_by": identifier_column(adult),
            "fractions": list(config.split_fractions),
            "seed": config.random_seed,
            "preprocessing_fit_partition": "train",
            "deviation_reference_partition": "train",
        },
        "config": config.as_dict(),
        "run_manifest": {
            "package_versions": _package_versions(),
            "random_seed": config.random_seed,
            "backend": backend,
            "device": device_name,
            "training_seconds": round(training_seconds, 3),
            "epochs_completed": len(history),
            "checkpoint_every": config.checkpoint_every,
            "resumed_from_epoch": start_epoch or None,
            "split_index_file": "splits.npz",
            "determinism_note": (
                "Seeded NumPy/torch generators, CPU by default. The torch backend on "
                "Apple Silicon mps is not bit-for-bit reproducible; see "
                "https://pytorch.org/docs/stable/notes/randomness.html"
            ),
        },
        "training_history": history,
        "final_validation_loss": history[-1]["validation_loss"] if history else None,
        "holdout_reconstruction_mse": float(
            np.mean(reconstruction_error[splits["holdout"]])
        )
        if len(splits["holdout"])
        else None,
        "capabilities": capabilities,
        "score_distribution": {
            **reference,
            "latent_mean": latent_mean.tolist(),
            "latent_std": latent_std.tolist(),
            "reference_partition": "train",
            "combined_sorted": training_reference.tolist(),
            "percentile_thresholds": {
                "90": float(np.quantile(training_reference, 0.90)),
                "95": float(np.quantile(training_reference, 0.95)),
                "99": float(np.quantile(training_reference, 0.99)),
            },
        },
        "posthoc_association_checks": posthoc,
        "posthoc_association_note": (
            "Cross-sectional associations with already-present conditions on the "
            "holdout partition. These are NOT future-risk performance."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "MODEL_CARD.md").write_text(render_model_card(metadata))
    return metadata


def render_model_card(metadata: dict[str, Any]) -> str:
    """Generate a model card that states the honest scope of the artifact."""
    fingerprint = metadata.get("dataset_fingerprint") or {}
    posthoc_lines = []
    for name, payload in (metadata.get("posthoc_association_checks") or {}).items():
        if payload.get("status") == "cross_sectional_association_only":
            posthoc_lines.append(
                f"- `{name}`: AUROC {payload['auroc']:.3f}, AUPRC {payload['auprc']:.3f} "
                f"({payload['positives']} positives / {payload['negatives']} negatives) - "
                "cross-sectional association only."
            )
        else:
            posthoc_lines.append(
                f"- `{name}`: not evaluated ({payload.get('reason', 'gated')})."
            )
    horizons = "1 year (365d), 3 years (1095d), 5 years (1825d)"
    return f"""# Model card: {metadata['model_name']} ({metadata['run_label']} run)

- **Code version:** {metadata['code_version']}
- **Created:** {metadata['created_at']}
- **Backend / device:** {metadata['backend']} / {metadata['device']}
- **Output type:** {metadata['output_type']}
- **Dataset:** {fingerprint.get('name', 'unknown')} (sha256 `{fingerprint.get('sha256', 'unknown')}`)
- **Rows:** {metadata['training_rows']} train / {metadata['validation_rows']} validation / {metadata['holdout_rows']} holdout
- **Latent dimension:** {metadata['latent_dim']}

## What this model does

It learns a label-free representation of adult metabolic features and reports:

1. a **metabolic deviation score** (how unusual a profile is versus the training reference),
2. a **reference percentile** of that score,
3. a **{metadata['latent_dim']}-dimensional latent representation**.

## What this model does NOT do

- It does not diagnose cancer or diabetes.
- It does not estimate the probability of developing any disease.
- It is not validated for the intended future horizons ({horizons}); those heads are
  disabled because no horizon passes the 50-event / 50-non-event safety gate on the
  current cross-sectional data.
- Type 1 diabetes remains research-only: no autoantibodies, no approved genetics and
  no confirmatory C-peptide criteria exist in these files.

## Training

- No outcome label is used during encoder training. Label columns present in the file
  but excluded from inputs: {', '.join(metadata['label_columns_present_but_unused_in_training']) or 'none'}.
- Splits are participant-grouped and seeded (seed {metadata['split_policy']['seed']},
  fractions {metadata['split_policy']['fractions']}).
- Preprocessing and deviation reference statistics are fit on the training partition only.
- Final validation reconstruction loss: {metadata['final_validation_loss']}.
- Holdout reconstruction MSE: {metadata['holdout_reconstruction_mse']}.

## Post-hoc association checks (cross-sectional, holdout only)

{chr(10).join(posthoc_lines) if posthoc_lines else '- none evaluated'}

These numbers describe association with conditions that are **already present**. They
must never be presented as future-risk performance.

## Known limitations and blocker

The single blocking limitation for future-risk work is data: NHANES here is a repeated
cross-section with one observation per participant and no follow-up, and TCGA is
post-diagnosis. Linked incident-outcome follow-up (for example NHANES-linked mortality
or registry linkage, or an EHR cohort) is required before any horizon-based head can be
trained or reported.
"""


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
    """Cross-sectional association probes. Never future-risk performance."""
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
        if positives < MIN_ASSOCIATION_CASES or negatives < MIN_ASSOCIATION_CASES:
            results[name] = {
                "status": "not_evaluated",
                "positives": positives,
                "negatives": negatives,
                "reason": (
                    f"Fewer than {MIN_ASSOCIATION_CASES} cases in one class on the "
                    "holdout partition."
                ),
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
    """Score records against a trained artifact (deviation + representation only)."""
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
    combined = deviation_scores(reconstruction, latent_distance, distribution)
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