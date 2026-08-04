"""Meeting-safe MetaboGuard demonstration runner.

One command that, on current valid data:

1. validates the dataset (fail-closed: coding, denylist, capability gates),
2. runs a bounded smoke self-supervised training job (CPU, minutes),
3. runs the PCA + Isolation Forest unsupervised baselines on the same
   preprocessing and split boundaries,
4. scores one example record to show the API output shape,
5. writes everything to a timestamped, clearly versioned run directory with a
   human-readable summary.

Every output is labelled representation/deviation research only. Nothing here
produces or implies a future disease probability, and smoke runs are never
promoted to the artifact the API serves.

Usage::

    python run_meeting_demo.py                 # smoke (default, bounded)
    python run_meeting_demo.py --full          # full SSL configuration
    python run_meeting_demo.py --skip-baselines
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from baselines import run_baselines
from data_integrity import file_fingerprint, validate_dataset
from self_supervised import SSLConfig, score_records, train_self_supervised
from train_self_supervised import SMOKE_OVERRIDES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"
DEMO_ROOT = PROJECT_ROOT / "model_artifacts" / "demo_runs"

EXAMPLE_RECORD: dict[str, Any] = {
    "DEMO_RIDAGEYR": 61,
    "DEMO_RIAGENDR": 2,
    "DEMO_RIDRETH3": 3,
    "BMX_BMXBMI": 33.1,
    "BMX_BMXWAIST": 106.0,
    "GHB_LBXGH": 7.1,
    "GLU_LBXGLU": 141,
    "INS_LBXIN": 21.4,
    "TRIGLY_LBXTR": 210,
    "HDL_LBDHDD": 38,
    "TCHOL_LBXTC": 205,
    "HSCRP_LBXHSCRP": 4.2,
    "weight_loss_1yr_lb": 12,
}


def summarise(payload: dict[str, Any]) -> str:
    training = payload["training"]
    lines = [
        "# MetaboGuard demonstration run",
        "",
        f"- Run label: **{training['run_label']}**",
        f"- Generated: {payload['generated_at']}",
        f"- Dataset: `{payload['dataset']['name']}` (sha256 `{payload['dataset']['sha256'][:16]}...`)",
        f"- Data validation: **{payload['data_validation']['status']}**",
        f"- Backend / device: {training['backend']} / {training['device']}",
        f"- Artifact: `{training['artifact_dir']}`",
        "",
        "## Output type",
        "",
        "Metabolic **deviation score**, **reference percentile** and "
        f"**{training['latent_dim']}-dimensional latent representation**. "
        "No future disease probability is produced. Longitudinal heads are disabled "
        "because no horizon (1y/3y/5y) passes the 50-event / 50-non-event gate.",
        "",
        "## Training result",
        "",
        f"- Rows: {training['training_rows']} train / {training['validation_rows']} validation / {training['holdout_rows']} holdout",
        f"- Final validation reconstruction loss: {training['final_validation_loss']}",
        f"- Holdout reconstruction MSE: {training['holdout_reconstruction_mse']}",
        f"- Wall time: {training['seconds']:.1f}s",
        "",
        "## Cross-sectional association probes (holdout, NOT future risk)",
        "",
    ]
    for name, item in (training.get("posthoc_association_checks") or {}).items():
        if item.get("status") == "cross_sectional_association_only":
            lines.append(
                f"- `{name}`: AUROC {item['auroc']:.3f}, AUPRC {item['auprc']:.3f} "
                f"({item['positives']} prevalent positives / {item['negatives']} negatives)"
            )
        else:
            lines.append(f"- `{name}`: not evaluated - {item.get('reason')}")
    baselines = payload.get("baselines")
    lines += ["", "## Unsupervised baselines (same preprocessing and splits)", ""]
    if baselines:
        for name, item in baselines["baselines"].items():
            lines.append(
                f"- `{name}`: holdout reconstruction MSE "
                f"{item.get('holdout_reconstruction_mse', 'n/a')}"
            )
        for pair, item in baselines["agreement"].items():
            lines.append(
                f"- Agreement {pair}: Spearman "
                f"{item['spearman_rank_correlation_holdout']:.3f}, "
                f"top-5% flag Jaccard {item['top5pct_flag_jaccard']:.3f}"
            )
    else:
        lines.append("- skipped")
    example = payload.get("example_score")
    lines += ["", "## Example record output", ""]
    if example:
        lines += [
            f"- Deviation score: {example['metabolic_deviation_score']}",
            f"- Reference percentile: {example['reference_percentile']}",
            f"- Latent length: {len(example['latent_representation'])}",
            f"- Interpretation: {example['interpretation']}",
        ]
    lines += [
        "",
        "## Remaining work for full validity",
        "",
        "- Full SSL run (40 epochs, 50k rows) when demonstrating final numbers.",
        "- Linked incident-outcome follow-up data before any 1/3/5-year head is trained.",
        "- Clinician review of deviation-flag review workflow.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--full", action="store_true", help="Run the full SSL configuration.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--output-root", default=str(DEMO_ROOT))
    arguments = parser.parse_args()

    dataset = Path(arguments.dataset).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = "full" if arguments.full else "smoke"
    run_dir = Path(arguments.output_root) / f"{label}__{stamp}"
    artifact_dir = run_dir / "ssl_artifact"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Validating {dataset.name} ...")
    report = validate_dataset(dataset, strict=True)
    (run_dir / "data_integrity_report.json").write_text(json.dumps(report.as_dict(), indent=2))
    print(f"      status={report.as_dict()['status']} rows={report.row_counts['rows']}")

    config = SSLConfig(backend=arguments.backend, device=arguments.device)
    if not arguments.full:
        for key, value in SMOKE_OVERRIDES.items():
            setattr(config, key, value)
    else:
        config.run_label = "full"

    print(f"[2/5] Training self-supervised encoder ({label}, label-free) ...")
    frame = pd.read_csv(dataset, low_memory=False)
    fingerprint = file_fingerprint(dataset)
    started = time.perf_counter()
    metadata = train_self_supervised(frame, artifact_dir, config, dataset_fingerprint=fingerprint)
    elapsed = time.perf_counter() - started
    print(f"      backend={metadata['backend']} seconds={elapsed:.1f}")

    baselines: dict[str, Any] | None = None
    if not arguments.skip_baselines:
        print("[3/5] Running PCA and Isolation Forest baselines ...")
        baselines = run_baselines(
            dataset,
            run_dir / "baselines",
            config=config,
            ssl_artifact_dir=artifact_dir,
            max_train_rows=min(config.max_train_rows, 20_000),
        )
    else:
        print("[3/5] Baselines skipped.")

    print("[4/5] Scoring one example record ...")
    example = score_records(pd.DataFrame([EXAMPLE_RECORD]), artifact_dir)[0]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": fingerprint,
        "data_validation": report.as_dict(),
        "training": {
            "run_label": metadata["run_label"],
            "artifact_dir": str(artifact_dir),
            "backend": metadata["backend"],
            "device": metadata["device"],
            "latent_dim": metadata["latent_dim"],
            "training_rows": metadata["training_rows"],
            "validation_rows": metadata["validation_rows"],
            "holdout_rows": metadata["holdout_rows"],
            "final_validation_loss": metadata["final_validation_loss"],
            "holdout_reconstruction_mse": metadata["holdout_reconstruction_mse"],
            "posthoc_association_checks": metadata["posthoc_association_checks"],
            "seconds": elapsed,
        },
        "baselines": baselines,
        "example_score": example,
        "output_type": "representation_and_deviation_research_only",
        "not_produced": [
            "future cancer probability",
            "future diabetes probability",
            "diagnosis",
        ],
        "promoted_to_api_default": False,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "RUN_SUMMARY.md").write_text(summarise(payload))
    print(f"[5/5] Wrote {run_dir}")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "artifact_dir": str(artifact_dir),
                "summary_markdown": str(run_dir / "RUN_SUMMARY.md"),
                "output_type": payload["output_type"],
                "reminder": "Representation/deviation research only - not a disease prediction.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()