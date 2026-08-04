"""One command that produces the whole MetaboGuard research pass artifact.

Steps, in order, all fail-closed:

1. Data-integrity validation (coding, denylist, capability gates).
2. Structured data-reliability report with feature eligibility tiers.
3. Biomarker evidence-catalogue validation and provenance summary.
4. Label-free phenotype clustering on the frozen representation - twice: on all adult
   rows, and as a complete-case sensitivity analysis.
5. Accessible SVG charts plus the exact CSV/JSON numbers behind each chart.
6. A combined ``RESEARCH_SUMMARY.md`` written for a meeting audience.

Nothing here predicts disease, assigns a cancer site, or names a cluster after a disease.

Usage::

    python run_research_pass.py                       # default grid (a few minutes)
    python run_research_pass.py --quick                # smaller grid for a fast demo
    python run_research_pass.py --skip-clustering      # reliability + evidence only
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

# Matplotlib must not try to write into a read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import pandas as pd  # noqa: E402

from clustering import ClusterConfig, run_clustering  # noqa: E402
from data_integrity import validate_dataset  # noqa: E402
from data_reliability import build_reliability_report  # noqa: E402
from evidence_catalogue import load_catalogue  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "model_artifacts"
    / "metaboguard_ssl"
    / "meeting_2026-08-04"
    / "ssl_artifact"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "model_artifacts" / "research_runs"


# ---------------------------------------------------------------------------
# Charts (SVG, colour-blind safe palette, text labels on every series)
# ---------------------------------------------------------------------------

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _charts(reliability: dict[str, Any], clustering: dict[str, Any] | None, output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Feature coverage by eligibility tier.
    eligibility = reliability["feature_eligibility"]
    coverage = reliability["sections"]["coverage_and_missingness"]["coverage_overall"]
    rows = [
        {
            "feature": name,
            "coverage": coverage.get(name, 0.0),
            "tier": payload["tier"],
        }
        for name, payload in eligibility.items()
        if payload["tier"] in {"usable_now", "qualified_use"} and name in coverage
    ]
    frame = pd.DataFrame(rows).sort_values("coverage")
    frame.to_csv(output / "feature_coverage.csv", index=False)
    if not frame.empty:
        figure, axes = plt.subplots(figsize=(9, max(4, 0.28 * len(frame))))
        colours = ["#0072B2" if tier == "usable_now" else "#D55E00" for tier in frame["tier"]]
        axes.barh(frame["feature"], frame["coverage"] * 100, color=colours)
        axes.set_xlabel("Coverage among adults (%)")
        axes.set_title("Model-input coverage by eligibility tier")
        axes.axvline(50, color="#444444", linestyle="--", linewidth=1)
        axes.annotate(
            "50% usable_now threshold", xy=(50, 0), xytext=(52, 0.5), fontsize=8
        )
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
            plt.Rectangle((0, 0), 1, 1, color="#D55E00"),
        ]
        axes.legend(handles, ["usable_now", "qualified_use"], loc="lower right")
        figure.tight_layout()
        path = output / "feature_coverage.svg"
        figure.savefig(path, format="svg", metadata={"Title": "Feature coverage by tier"})
        plt.close(figure)
        written.append(str(path))

    if clustering is None:
        return written

    # 2. Cluster quality and stability per candidate.
    candidates = [item for item in clustering["candidates"] if item.get("status") == "evaluated"]
    if candidates:
        labels = [f"{item['method'][:6]}\nk={item['k']}" for item in candidates]
        silhouette = [item["train_metrics"].get("silhouette") or 0 for item in candidates]
        stability = [item["bootstrap_stability"].get("mean_ari") or 0 for item in candidates]
        figure, axes = plt.subplots(figsize=(max(7, 0.9 * len(candidates)), 4.5))
        positions = range(len(candidates))
        axes.plot(positions, silhouette, marker="o", color=PALETTE[0], label="Silhouette")
        axes.plot(positions, stability, marker="s", color=PALETTE[1], label="Bootstrap ARI")
        axes.axhline(0.15, color=PALETTE[0], linestyle=":", linewidth=1)
        axes.axhline(0.60, color=PALETTE[1], linestyle=":", linewidth=1)
        axes.set_xticks(list(positions))
        axes.set_xticklabels(labels, fontsize=8)
        axes.set_ylim(0, 1.05)
        axes.set_ylabel("Score")
        axes.set_title("Cluster quality and resample stability (dotted lines = gates)")
        axes.legend()
        figure.tight_layout()
        path = output / "cluster_quality_stability.svg"
        figure.savefig(path, format="svg", metadata={"Title": "Cluster quality and stability"})
        plt.close(figure)
        written.append(str(path))

    # 3. Negative controls heat-strip.
    control_names = sorted(
        {
            name
            for item in candidates
            for name in item.get("negative_controls", {}).get("controls", {})
        }
    )
    if control_names:
        matrix = [
            [
                item["negative_controls"]["controls"].get(name, {}).get("value", 0) or 0
                for name in control_names
            ]
            for item in candidates
        ]
        figure, axes = plt.subplots(figsize=(1.6 * len(control_names) + 3, 0.5 * len(candidates) + 2.5))
        image = axes.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes.set_xticks(range(len(control_names)))
        axes.set_xticklabels([name.replace("_", "\n") for name in control_names], fontsize=7)
        axes.set_yticks(range(len(candidates)))
        axes.set_yticklabels(
            [f"{item['method']} k={item['k']}" for item in candidates], fontsize=8
        )
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axes.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.6 else "black",
                )
        axes.set_title("Negative controls (>0.30 = data artefact)")
        figure.colorbar(image, ax=axes, label="Association strength")
        figure.tight_layout()
        path = output / "negative_controls.svg"
        figure.savefig(path, format="svg", metadata={"Title": "Negative controls"})
        plt.close(figure)
        written.append(str(path))

    # 4. Projection scatter, only when a solution was selected.
    projection = clustering.get("projection")
    if projection:
        points = pd.DataFrame(projection["points"])
        figure, axes = plt.subplots(figsize=(6, 5))
        for index, (cluster, group) in enumerate(points.groupby("cluster")):
            axes.scatter(
                group["x"],
                group["y"],
                s=8,
                alpha=0.6,
                color=PALETTE[index % len(PALETTE)],
                label=f"phenotype {cluster}",
            )
        axes.set_xlabel("PC1 of clustering space")
        axes.set_ylabel("PC2 of clustering space")
        axes.set_title("Phenotype projection (research only; not disease groups)")
        axes.legend(fontsize=8)
        figure.tight_layout()
        path = output / "phenotype_projection.svg"
        figure.savefig(path, format="svg", metadata={"Title": "Phenotype projection"})
        plt.close(figure)
        written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summary_markdown(payload: dict[str, Any]) -> str:
    reliability = payload["reliability"]
    evidence = payload["evidence"]
    lines = [
        "# MetaboGuard research pass",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Dataset: `{payload['dataset']['name']}` (sha256 `{payload['dataset']['sha256'][:16]}...`)",
        f"- Frozen encoder artifact: `{payload['artifact']}`",
        f"- Data integrity: **{payload['integrity_status']}**",
        f"- Data reliability: **{reliability['status']}**",
        f"- Evidence catalogue: **{evidence['status']}** "
        f"({evidence['doctor_facing_ready']} of {evidence['entry_count']} rows clinician-ready)",
        "",
        "## What this pass is",
        "",
        "Exploratory, label-free discovery of patient/metabolic **phenotypes**, plus a "
        "structured reliability audit of the inputs and a provenance-checked evidence "
        "catalogue. Nothing here predicts future disease, assigns a cancer site, or names a "
        "cluster after a disease. Early detection is treated as a **panel and "
        "feature-interaction** problem; the claim that cancers have no specific markers is "
        "false and is not made anywhere in this project.",
        "",
        "## Feature eligibility tiers",
        "",
        "| Tier | Count | Features |",
        "| --- | --- | --- |",
    ]
    for tier, features in reliability["tiers"].items():
        shown = ", ".join(f"`{name}`" for name in features[:8])
        if len(features) > 8:
            shown += f", … (+{len(features) - 8})"
        lines.append(f"| `{tier}` | {len(features)} | {shown or '—'} |")
    drift = reliability["sections"]["assay_cycle_drift"]
    lines += [
        "",
        f"- Cycle-level drift flags: {drift.get('level_drift_features') or 'none'}",
        f"- Cycle availability gaps: {drift.get('availability_gap_features') or 'none'}",
        f"- Survey weights applied in modelling: "
        f"{reliability['sections']['survey_weights']['weights_applied_in_modelling']}",
        "",
        "## Clustering result",
        "",
    ]
    for name, result in payload.get("clustering", {}).items():
        if result is None:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Status: **{result['status']}**")
        lines.append(f"- Space: `{result['space']}` ({result['space_dimension']} dimensions)")
        lines.append(f"- Split source: {result['split_source']}")
        if result["status"] == "no_stable_clusters":
            lines.append(f"- Abstain reason: {result['abstain_reason']}")
            lines.append("- Gate failures per candidate:")
            for candidate, failures in (result.get("gate_failure_summary") or {}).items():
                lines.append(f"  - `{candidate}`: {', '.join(failures) or 'none'}")
        else:
            selected = result["selected"]
            lines.append(
                f"- Selected: **{selected['method']}, k={selected['k']}** "
                f"(silhouette {selected['train_metrics']['silhouette']}, "
                f"bootstrap ARI {selected['bootstrap_stability']['mean_ari']})"
            )
            lines.append(
                "- Negative controls: "
                + ", ".join(
                    f"{key} {item['value']}"
                    for key, item in selected["negative_controls"]["controls"].items()
                )
            )
            for cluster in result["characterisation"]["clusters"]:
                panel = ", ".join(
                    f"{item['feature']} {item['direction']}"
                    for item in cluster["top_distinguishing_panel"][:4]
                )
                lines.append(
                    f"  - `{cluster['cluster_id']}`: {cluster['rows']} rows "
                    f"({cluster['share_of_partition']:.1%}) — panel: {panel}"
                )
        lines.append("")
    lines += [
        "## Interpretation classes used in every clinician-facing statement",
        "",
        "| Class | Meaning |",
        "| --- | --- |",
        "| `data observation` | Measured directly in the file (coverage, drift, counts). |",
        "| `model association` | Produced by our model on our sample; not validated, not causal. |",
        "| `published evidence` | Taken from a catalogued source with URL, design and grade. |",
        "| `causal claim not established` | The default for every mechanism statement. |",
        "",
        "## Scientific blockers unchanged by this pass",
        "",
        "1. **Early-stage detection needs longitudinal data.** No follow-up time, no incident "
        "outcome and no stage information exists in these files, so clinical value in "
        "Prof. Helmout's sense cannot be demonstrated here.",
        "2. **Cancer site cannot be assigned.** Site comes from a self-reported multi-select "
        "item with 19 prevalent pancreatic cases; site outputs stay disabled.",
        "3. **Survey design is not modelled.** Without MEC weights and PSU/strata variance "
        "estimation, nothing here is a population estimate.",
        "",
        "## Files in this run",
        "",
    ]
    for name in payload.get("files", []):
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--quick", action="store_true", help="Smaller clustering grid.")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    arguments = parser.parse_args()

    dataset = Path(arguments.dataset).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(arguments.output_root) / f"research__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/5] Data integrity ...")
    integrity = validate_dataset(dataset, strict=True)
    (run_dir / "data_integrity_report.json").write_text(
        json.dumps(integrity.as_dict(), indent=2)
    )

    print("[2/5] Data reliability ...")
    reliability = build_reliability_report(dataset, strict=True).as_dict()
    (run_dir / "data_reliability_report.json").write_text(json.dumps(reliability, indent=2))
    print(
        "      tiers: "
        + ", ".join(f"{name}={len(items)}" for name, items in reliability["tiers"].items())
    )

    print("[3/5] Evidence catalogue ...")
    catalogue = load_catalogue(strict=True)
    evidence = catalogue.summary()
    (run_dir / "evidence_catalogue_report.json").write_text(json.dumps(evidence, indent=2))

    clustering_results: dict[str, Any] = {}
    if not arguments.skip_clustering:
        grid = (2, 3, 4) if arguments.quick else (2, 3, 4, 5, 6)
        methods = ("kmeans",) if arguments.quick else ("kmeans", "gaussian_mixture")
        rounds = 5 if arguments.quick else 8
        fit_rows = 4_000 if arguments.quick else 6_000
        for name, complete_cases in (("all_adults", False), ("complete_cases", True)):
            print(f"[4/5] Clustering ({name}) ...")
            config = ClusterConfig(
                k_values=grid,
                methods=methods,
                bootstrap_rounds=rounds,
                max_fit_rows=fit_rows,
                restrict_to_complete_cases=complete_cases,
            )
            clustering_results[name] = run_clustering(
                dataset, arguments.artifact, run_dir / f"clustering_{name}", config
            )
            print(f"      status={clustering_results[name]['status']}")
    else:
        print("[4/5] Clustering skipped.")

    charts: list[str] = []
    if not arguments.skip_charts:
        print("[5/5] Charts ...")
        primary = clustering_results.get("complete_cases") or clustering_results.get("all_adults")
        charts = _charts(reliability, primary, run_dir / "charts")

    payload = {
        "run_type": "metaboguard_research_pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "seconds": round(time.perf_counter() - started, 1),
        "dataset": reliability["sections"]["provenance"]["fingerprint"],
        "artifact": arguments.artifact,
        "integrity_status": integrity.as_dict()["status"],
        "reliability": reliability,
        "evidence": evidence,
        "clustering": {
            name: {
                key: value
                for key, value in result.items()
                if key != "projection"  # projection is persisted separately as CSV
            }
            for name, result in clustering_results.items()
        },
        "charts": charts,
        "files": sorted(
            str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
        ),
        "output_type": "research_only_reliability_and_phenotype_exploration",
        "not_produced": [
            "future disease probability",
            "cancer site assignment",
            "diagnosis",
            "causal claim",
        ],
    }
    (run_dir / "research_pass.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "RESEARCH_SUMMARY.md").write_text(_summary_markdown(payload))
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "summary": str(run_dir / "RESEARCH_SUMMARY.md"),
                "reliability_status": reliability["status"],
                "evidence_status": evidence["status"],
                "clustering_status": {
                    name: result["status"] for name, result in clustering_results.items()
                },
                "charts": charts,
                "seconds": payload["seconds"],
                "reminder": (
                    "Research only: reliability audit and exploratory phenotypes. "
                    "No future risk, no cancer site, no diagnosis."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()