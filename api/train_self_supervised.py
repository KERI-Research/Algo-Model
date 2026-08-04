"""Train the MetaboGuard self-supervised metabolic representation model.

Every run is data-validated first (fail-closed) and writes a timestamped,
versioned artifact directory plus a model card. Smoke runs are never promoted to
the artifact the API serves.

Examples
--------
Validate + bounded smoke run (minutes, CPU, no PyTorch required)::

    python train_self_supervised.py --smoke

Full run with the committed configuration::

    python train_self_supervised.py --config configs/ssl_full.json --promote
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd

from data_integrity import file_fingerprint, validate_dataset
from self_supervised import SSLConfig, train_self_supervised

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = PROJECT_ROOT / "model_artifacts" / "metaboguard_ssl"
RUN_ROOT = ARTIFACT_ROOT / "runs"
POINTER_PATH = ARTIFACT_ROOT / "CURRENT.json"

#: Bounded configuration used for meeting demonstrations.
SMOKE_OVERRIDES = {
    "epochs": 3,
    "max_train_rows": 6_000,
    "batch_size": 256,
    "patience": 2,
    "checkpoint_every": 1,
    "run_label": "smoke",
}


def build_config(arguments: argparse.Namespace) -> SSLConfig:
    config = (
        SSLConfig.from_file(arguments.config) if arguments.config else SSLConfig()
    )
    if arguments.smoke:
        for key, value in SMOKE_OVERRIDES.items():
            setattr(config, key, value)
    for key in (
        "epochs",
        "latent_dim",
        "max_train_rows",
        "batch_size",
        "backend",
        "device",
        "random_seed",
        "run_label",
    ):
        value = getattr(arguments, key)
        if value is not None:
            setattr(config, key, value)
    if arguments.resume:
        config.resume = True
    return config


def resolve_output_dir(arguments: argparse.Namespace, dataset: Path, run_label: str) -> Path:
    if arguments.output_dir:
        return Path(arguments.output_dir).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RUN_ROOT / f"{dataset.stem}__{run_label}__{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"))
    parser.add_argument("--config", default=None, help="JSON file with SSLConfig fields.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Bounded demonstration run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Point model_artifacts/metaboguard_ssl/CURRENT.json at this run (blocked for smoke runs).",
    )
    arguments = parser.parse_args()

    dataset = Path(arguments.dataset).resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    # Fail closed: invalidated files, denylist leaks and coding errors stop here.
    report = validate_dataset(dataset, strict=True)

    config = build_config(arguments)
    output = resolve_output_dir(arguments, dataset, config.run_label)
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset, low_memory=False)
    fingerprint = file_fingerprint(dataset)
    metadata = train_self_supervised(frame, output, config, dataset_fingerprint=fingerprint)
    (output / "data_integrity_report.json").write_text(json.dumps(report.as_dict(), indent=2))

    promoted = False
    if arguments.promote:
        if config.run_label == "smoke":
            print("[gate] Refusing to promote a smoke run to CURRENT.json.")
        else:
            POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            POINTER_PATH.write_text(
                json.dumps(
                    {
                        "artifact_dir": str(output),
                        "run_label": config.run_label,
                        "model_name": metadata["model_name"],
                        "code_version": metadata["code_version"],
                        "dataset_sha256": fingerprint["sha256"],
                        "promoted_at": datetime.now(UTC).isoformat(),
                        "output_type": metadata["output_type"],
                    },
                    indent=2,
                )
            )
            promoted = True

    print(
        json.dumps(
            {
                "artifact": str(output),
                "run_label": config.run_label,
                "backend": metadata["backend"],
                "device": metadata["device"],
                "model_name": metadata["model_name"],
                "dataset": fingerprint["name"],
                "dataset_sha256": fingerprint["sha256"],
                "training_rows": metadata["training_rows"],
                "validation_rows": metadata["validation_rows"],
                "holdout_rows": metadata["holdout_rows"],
                "final_validation_loss": metadata["final_validation_loss"],
                "holdout_reconstruction_mse": metadata["holdout_reconstruction_mse"],
                "capabilities": metadata["capabilities"],
                "posthoc_association_checks": metadata["posthoc_association_checks"],
                "promoted_to_current": promoted,
                "output_type": metadata["output_type"],
                "reminder": (
                    "Deviation scores and latent representations only. "
                    "No future disease probability is produced or implied."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()