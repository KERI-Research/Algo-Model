"""Generate corrected v2 MetaboGuard documentation visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "nhanes_multicycle_v2.csv"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

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
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
    })


def outcome_audit(frame: pd.DataFrame) -> None:
    slot_columns = ["MCQ_MCQ230A", "MCQ_MCQ230B", "MCQ_MCQ230C", "MCQ_MCQ230D"]
    slots = frame[slot_columns].apply(pd.to_numeric, errors="coerce")
    values = [int((slots == 39).any(axis=1).sum()), int((slots == 29).any(axis=1).sum())]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    bars = ax.bar(
        ["Code 39\nOther cancer", "Code 29\nPancreatic cancer"],
        values,
        color=[RUST, TEAL],
        width=0.55,
    )
    ax.set_title(
        "Official codebook verification changed the modelling cohort\n"
        "Code 39 is Other; code 29 is Pancreas",
        loc="left",
    )
    ax.set_ylabel("Participants reporting the site code")
    ax.set_ylim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:,}",
            ha="center",
            fontweight="bold",
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ASSETS / "outcome-code-audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def variable_coverage(frame: pd.DataFrame) -> None:
    columns = {
        "Haemoglobin": "CBC_LBXHGB",
        "Platelets": "CBC_LBXPLTSI",
        "ALT": "BIOPRO_LBXSATSI",
        "Alkaline phosphatase": "BIOPRO_LBXSAPSI",
        "Creatinine": "BIOPRO_LBXSCR",
        "Nonlinear HbA1c terms": "hba1c_squared",
        "Smoking status": "smoking_status",
        "Alcohol status": "alcohol_status",
        "Average drinks/day": "average_drinks_per_day",
    }
    coverage = pd.Series({
        label: frame[column].notna().mean() * 100
        for label, column in columns.items()
    }).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    bars = ax.barh(coverage.index, coverage.values, color=TEAL)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pooled participants with a measured/derived value (%)")
    ax.set_title(
        "Priority A variables have useful cross-cycle coverage\n"
        "Coverage does not solve the insufficient pancreatic-cancer event count",
        loc="left",
    )
    for bar, value in zip(bars, coverage.values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ASSETS / "priority-a-variable-coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_title(
        "MetaboGuard v2: corrected labels and safety-gated modelling",
        loc="left", fontsize=16, pad=14,
    )
    boxes = [
        (0.3, "NHANES v2", "107,622 participants"),
        (2.9, "Verified outcome", "19 pancreatic cases"),
        (5.5, "Diabetic cohort", "7 cases; 6 usable"),
        (8.1, "NODM-PC target", "2 cases within 3 years"),
        (10.7, "Safety gate", "No model export"),
    ]
    for index, (x, title, subtitle) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 2.0), 2.1, 1.4,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#FBFBF9",
            edgecolor=TEAL if index in (0, 4) else GRID,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(x + 1.05, 2.92, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + 1.05, 2.38, subtitle, ha="center", va="center",
                fontsize=9, color=MUTED)
        if index < len(boxes) - 1:
            ax.annotate(
                "", xy=(boxes[index + 1][0] - 0.12, 2.7),
                xytext=(x + 2.22, 2.7),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5),
            )
    ax.text(
        6.5, 0.75,
        "Priority A variables are implemented, but a larger incident cohort is required.",
        ha="center", va="center", fontsize=10, color=MUTED,
    )
    fig.savefig(ASSETS / "metaboguard-v2-pipeline-overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    usecols = [
        "MCQ_MCQ230A", "MCQ_MCQ230B", "MCQ_MCQ230C", "MCQ_MCQ230D",
        "CBC_LBXHGB", "CBC_LBXPLTSI", "BIOPRO_LBXSATSI",
        "BIOPRO_LBXSAPSI", "BIOPRO_LBXSCR", "hba1c_squared",
        "smoking_status", "alcohol_status", "average_drinks_per_day",
    ]
    frame = pd.read_csv(DATA, usecols=usecols, low_memory=False)
    outcome_audit(frame)
    variable_coverage(frame)
    pipeline_diagram()
    for path in sorted(ASSETS.glob("*v2*.png")):
        print(path)
