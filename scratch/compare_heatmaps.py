import os
import hashlib
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

STYLE = {
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#0f3460",   "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",      "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",       "grid.color": "#0f3460",
    "grid.linestyle": "--",        "grid.alpha": 0.4,
}

def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v

def fig_diversity_heatmap(div_csv, out):
    df = pd.read_csv(div_csv, index_col=0)
    data = df.to_numpy(dtype=float)
    labels = list(df.columns)

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#e15759","#f28e2b","#59a14f"])
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(data[i,j]) < 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Agent Pairwise Correlation\n(Low = Complementary, RQ3)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# Generate both
csv_4x4 = Path("results/multi_agent/agent_diversity_matrix.csv")
csv_6x6 = Path("archive/results_backup_20260514_142108/agent_diversity_matrix.csv")

fig_diversity_heatmap(csv_4x4, "scratch/heatmap_4x4.png")
fig_diversity_heatmap(csv_6x6, "scratch/heatmap_6x6.png")

def get_md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("scratch/heatmap_4x4.png MD5:", get_md5("scratch/heatmap_4x4.png"), "size:", os.path.getsize("scratch/heatmap_4x4.png"))
print("scratch/heatmap_6x6.png MD5:", get_md5("scratch/heatmap_6x6.png"), "size:", os.path.getsize("scratch/heatmap_6x6.png"))
print("latex_assets/thesis_agent_diversity_heatmap.png MD5:", get_md5("latex_assets/thesis_agent_diversity_heatmap.png"), "size:", os.path.getsize("latex_assets/thesis_agent_diversity_heatmap.png"))
