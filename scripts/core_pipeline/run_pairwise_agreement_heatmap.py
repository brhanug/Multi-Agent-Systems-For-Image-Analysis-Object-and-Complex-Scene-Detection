#!/usr/bin/env python3
"""
run_pairwise_agreement_heatmap.py
-----------------------------------
Regenerates the "Pairwise Inter-Agent Consensus Agreement Matrix" figure
cited in the thesis and defense slides.

Replaces a stale figure (pairwise_agreement_heatmap_v8.png) that showed
correlations between OWL-ViT, DINO, YOLO-Teacher, YOLO-Student, and
Florence-2 -- models from an early architecture iteration, none of which
are part of the deployed DCV-MAC Tier 1 ensemble. The caption had never
been updated to match, so the figure silently drifted out of sync with the
actual architecture description around it.

This script computes the real Pearson correlation matrix between the four
Tier 1 agents actually deployed (spatial grounding, scene classification,
VQA, grounded OCR) on the full deployment corpus C0, and regenerates the
figure fresh each run so it can't go stale unnoticed again.

Outputs:
  results/figures/pairwise_agreement_heatmap_v9.png
  latex_assets/pairwise_agreement_heatmap_v9.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_FIGURES = BASE / "results/figures/pairwise_agreement_heatmap_v9.png"
OUT_LATEX_ASSETS = BASE / "latex_assets/pairwise_agreement_heatmap_v9.png"

COLS = ["existing_pipeline_agent", "scene_agent", "vlm_agent", "document_agent"]
LABELS = ["YOLOv11+SAM\n(Spatial)", "SigLIP\n(Scene)", "LLaVA-OneVision\n(VQA)", "Kosmos-2.5\n(OCR)"]


def main() -> None:
    df = pd.read_csv(SCORES_CSV)
    corr = df[COLS].corr(method="pearson").to_numpy()
    print(f"Pairwise agreement matrix (Pearson r, n={len(df)}):")
    print(pd.DataFrame(np.round(corr, 4), index=LABELS, columns=LABELS))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = corr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.6 else "black", fontsize=11)
    ax.set_title(f"Correlation between the deployed agent signals\n"
                 f"(Pearson $r$, $C_0$, $n={len(df):,}$)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()

    OUT_FIGURES.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIGURES, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_LATEX_ASSETS, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Wrote {OUT_FIGURES}")
    print(f"✅ Wrote {OUT_LATEX_ASSETS}")


if __name__ == "__main__":
    main()
