"""Generate MetaboGuard prevention-model documentation diagrams."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BG = "#F7F6F2"
TEAL = "#20808D"
DARK = "#28251D"
MUTED = "#7A7974"
GRID = "#D4D1CA"


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(
        "MetaboGuard hybrid self-supervised prevention architecture",
        loc="left", fontsize=18, fontweight="bold", color=DARK,
    )
    boxes = [
        (0.3, "Pre-diagnosis inputs", "Metabolic, CBC,\nbiochemistry, lifestyle"),
        (3.2, "Denoising encoder", "Masked reconstruction\nwithout disease labels"),
        (6.1, "Shared representation", "16 latent dimensions\n+ deviation percentile"),
        (9.0, "Clinician review", "Unusual profile +\ntop deviating features"),
        (11.9, "Future risk heads", "Only with longitudinal\nincident outcomes"),
    ]
    for index, (x, title, subtitle) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 2.2), 2.35, 1.55,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#FBFBF9",
            edgecolor=TEAL if index in (1, 3) else GRID,
            linewidth=2,
        )
        ax.add_patch(patch)
        ax.text(x + 1.175, 3.22, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=DARK)
        ax.text(x + 1.175, 2.62, subtitle, ha="center", va="center",
                fontsize=9.2, color=MUTED)
        if index < len(boxes) - 1:
            ax.annotate(
                "", xy=(boxes[index + 1][0] - 0.12, 2.98),
                xytext=(x + 2.47, 2.98),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6),
            )
    ax.text(
        7, 0.8,
        "Current NHANES capability: representation and metabolic deviation only. "
        "No diagnosis or future-risk probability.",
        ha="center", fontsize=10.5, color=MUTED,
    )
    fig.savefig(
        ASSETS / "metaboguard-ssl-architecture.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    print(ASSETS / "metaboguard-ssl-architecture.png")
