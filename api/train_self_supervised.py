"""Train the MetaboGuard self-supervised metabolic representation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from self_supervised import SSLConfig, train_self_supervised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="../data/nhanes_multicycle_v2.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    root = Path(__file__).resolve().parent.parent
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "model_artifacts" / "metaboguard_ssl" / dataset.stem
    )
    frame = pd.read_csv(dataset, low_memory=False)
    metadata = train_self_supervised(
        frame,
        output,
        SSLConfig(
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            max_train_rows=args.max_train_rows,
        ),
    )
    print(json.dumps({
        "artifact": str(output),
        "model_name": metadata["model_name"],
        "training_rows": metadata["training_rows"],
        "capabilities": metadata["capabilities"],
        "posthoc_association_checks": metadata["posthoc_association_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
