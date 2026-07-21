from __future__ import annotations

import argparse
import json
from pathlib import Path

from biomarker import train_biomarker_model
from fetch_nhanes import ensure_nhanes_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the local NHANES biomarker discovery model and write a versioned artifact.",
    )
    parser.add_argument(
        "--dataset",
        default="nhanes_merged.csv",
        help="Dataset filename or path. Defaults to nhanes_merged.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if an artifact already exists.",
    )
    return parser


def resolve_training_path(dataset_name: str) -> Path:
    dataset_path = Path(dataset_name)
    if dataset_path.exists() and dataset_path.is_file():
        return dataset_path

    if dataset_path.name == "nhanes_merged.csv":
        return ensure_nhanes_dataset()

    api_dir = Path(__file__).resolve().parent
    project_root = api_dir.parent
    candidates = [
        project_root / "data" / dataset_path.name,
        api_dir / "nhanes_data" / dataset_path.name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Unable to resolve dataset: {dataset_name}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset_path = resolve_training_path(args.dataset)
    artifact = train_biomarker_model(str(dataset_path), force=args.force)

    summary = {
        "dataset": str(dataset_path),
        "model": artifact["model_name"],
        "artifact_version": artifact["artifact_version"],
        "created_at": artifact["created_at"],
        "metrics": artifact["metrics"],
        "benchmarks": artifact.get("benchmarks", {}),
        "cohort_summary": artifact["cohort_summary"],
        "top_biomarkers": artifact["biomarker_ranking"][:5],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()