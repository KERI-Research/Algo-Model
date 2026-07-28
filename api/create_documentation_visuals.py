"""Generate accessible documentation visuals for DiaPan."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
REPORT = json.loads((ROOT / "cycle_holdout_validation.json").read_text())

TEAL = "#20808D"
RUST = "#A84B2F"
DARK = "#28251D"
MUTED = "#7A7974"
GRID = "#D4D1CA"
BG = "#F7F6F2"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.grid": False,
    })


def performance_comparison() -> None:
    random = {"AUROC": 0.641317, "AUPRC": 0.134717}
    temporal = REPORT["pooled_out_of_cycle"]["clinical_only"]["diapan_xgboost_v1"]
    values = {
        "AUROC": [random["AUROC"], temporal["auroc"]],
        "AUPRC": [random["AUPRC"], temporal["auprc"]],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    fig.suptitle(
        "Random splitting overstates precision-recall performance",
        fontsize=17,
        fontweight="bold",
    )
    for ax, (metric, pair) in zip(axes, values.items()):
        bars = ax.bar(
            ["Random 80/20 split", "Cycle-held-out"],
            pair,
            color=[RUST, TEAL],
            width=0.6,
        )
        ax.set_ylim(0, max(pair) * 1.35)
        ax.set_title(
            f"{metric}\n"
            + ("Higher is better; 0.5 = random ranking" if metric == "AUROC"
               else "Higher is better; prevalence baseline = 0.0063"),
            loc="left",
            fontsize=13,
        )
        for bar, value in zip(bars, pair):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(pair) * 0.035,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=0)
    note = textwrap.fill(
        "Random splitting mixes survey eras and can overstate screening yield. "
        "Cycle-held-out validation trains without the test cycle.",
        105,
    )
    fig.text(0.01, -0.10, note, fontsize=9, color=MUTED)
    fig.savefig(ASSETS / "random-vs-temporal-performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def cycle_stability() -> None:
    folds = [
        row for row in REPORT["folds"]
        if row["variant"] == "clinical_only" and row["model"] == "diapan_xgboost_v1"
    ]
    cycles = [row["held_out_cycle"] for row in folds]
    auroc = [row["auroc"] for row in folds]
    lift = [row["auprc_lift"] for row in folds]
    positives = [row["test_positives"] for row in folds]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
    x = np.arange(len(cycles))
    axes[0].plot(x, auroc, marker="o", color=TEAL, linewidth=2)
    axes[0].axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_ylim(0.35, 1.0)
    axes[0].set_ylabel("AUROC (higher is better)")
    axes[0].set_title(
        "Performance varies materially across survey cycles\n"
        "Clinical-only XGBoost; dashed line is random ranking",
        loc="left",
    )
    for i, (value, count) in enumerate(zip(auroc, positives)):
        axes[0].annotate(
            f"{value:.2f}\n({count} +)",
            (i, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(x, lift, color=TEAL, width=0.65)
    axes[1].axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_ylabel("AUPRC lift over prevalence")
    axes[1].set_title(
        "Screening yield is unstable when cycles contain few positive cases\n"
        "Higher is better; 1× means no improvement over prevalence",
        loc="left",
        fontsize=13,
    )
    axes[1].set_xticks(x, cycles, rotation=35, ha="right")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(ASSETS / "cycle-held-out-stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_title(
        "DiaPan research pipeline: from open cohorts to comparable benchmark outputs",
        loc="left",
        fontsize=16,
        pad=14,
    )
    boxes = [
        (0.3, 2.0, 2.1, 1.4, "NHANES\n1999–2020", "Metabolic + survey data"),
        (2.9, 2.0, 2.1, 1.4, "Harmonisation", "Cycle aliases, units,\nlabels, missingness"),
        (5.5, 2.0, 2.1, 1.4, "Diabetic cohort", "6,473 model rows\n41 positives"),
        (8.1, 2.0, 2.1, 1.4, "Temporal benchmark", "Hold out one cycle\nat a time"),
        (10.7, 2.0, 2.1, 1.4, "Research outputs", "Risk score, metrics,\nmodel card"),
    ]
    for i, (x, y, w, h, title, subtitle) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#FBFBF9",
            edgecolor=TEAL if i in (0, 4) else GRID,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=9, color=MUTED)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[i + 1][0] - 0.12, y + h / 2),
                xytext=(x + w + 0.12, y + h / 2),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5),
            )
    ax.text(
        6.5, 0.75,
        "TCGA-CDR remains a complementary prognosis dataset; it is not used as "
        "pre-diagnosis metabolic training data.",
        ha="center", va="center", fontsize=10, color=MUTED,
    )
    fig.savefig(ASSETS / "diapan-pipeline-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    performance_comparison()
    cycle_stability()
    pipeline_diagram()
    print("\n".join(str(path) for path in sorted(ASSETS.glob("*.png"))))
