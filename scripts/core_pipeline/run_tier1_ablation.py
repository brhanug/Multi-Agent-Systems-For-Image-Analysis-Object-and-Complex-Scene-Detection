#!/usr/bin/env python3
"""
run_tier1_ablation.py
-----------------------
Replaces run_6_agent_ablation.py, which is deleted alongside this script's
introduction. That script fabricated 3 of its 6 "agents" via a hash-based
pseudo-random formula (code comment: "Simulate agent 1, 2, 3 ... Ultra fast
pseudo-random based on hash") and labeled its rows using Tier-2 role names
(e.g. "Remove Agent 4 (Cross-Archive Linker)") that did not match the data
actually loaded for that row (demographic_profile.csv / social composition
score -- an unrelated agent under a different, older numbering scheme).
It was wired into the reproducible pipeline, so it silently regenerated
fabricated-looking-real numbers on every run.

This script performs the ablation the thesis's own architecture description
actually supports: per the "Architectural Decoupling Rule" (Chapter 3),
only the four Tier 1 agents participate in the core SAA disagreement /
uncertainty calculation -- Tier 2 (Contradiction, Cross-Archive Linker,
Grounded OCR curation) and the auxiliary prototype tooling (Temporal
Historian, Bayesian Ranker, Frontier Auditor) explicitly do not. A leave-
one-out ablation is therefore only meaningful, and only claimed, over the
four Tier 1 agents.

Outputs:
  results/multi_agent/tier1_ablation.csv
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
OUT_CSV = BASE / "results/multi_agent/tier1_ablation.csv"
OUT_FIG = BASE / "latex_assets/tier1_ablation_figure.png"

AGENT_COLS = {
    "existing_pipeline_agent": "Spatial Grounding (YOLOv11+SAM)",
    "scene_agent": "Scene Classification (SigLIP)",
    "vlm_agent": "VQA (LLaVA-OneVision)",
    "document_agent": "Grounded OCR (Kosmos-2.5)",
}


def main() -> None:
    df = pd.read_csv(SCORES_CSV)
    cols = list(AGENT_COLS.keys())

    def mean_sigma(leave_out: str | None) -> float:
        active = [c for c in cols if c != leave_out]
        return float(df[active].std(axis=1).mean())

    rows = [("Full Tier 1 Ensemble (4 agents)", mean_sigma(None))]
    for col in cols:
        rows.append((f"Remove {AGENT_COLS[col]}", mean_sigma(col)))

    out = pd.DataFrame(rows, columns=["Ablated_Component", "Mean_Uncertainty_Sigma"])
    out["Mean_Uncertainty_Sigma"] = out["Mean_Uncertainty_Sigma"].round(4)
    out["Delta_vs_Full"] = (out["Mean_Uncertainty_Sigma"] - out.loc[0, "Mean_Uncertainty_Sigma"]).round(4)
    print(out.to_string(index=False))
    out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Wrote {OUT_CSV}")

    # ---- figure: mean per-image std across the four Tier 1 MODELS (Object/Scene/VLM/OCR).
    # NOTE: this is sigma_model, NOT U_SAA. U_SAA (run_clean_holdout_evaluation.py:82) is the std
    # over [object, agreement, scene, vlm] -- it substitutes the derived Agreement signal for OCR.
    # The two share no more than three columns and must not be conflated. ----
    full_sigma = out.loc[0, "Mean_Uncertainty_Sigma"]
    labels = [
        "Full Tier 1\nEnsemble",
        "Remove\nSpatial Grounding",
        "Remove\nScene Classification",
        "Remove\nVQA",
        "Remove\nGrounded OCR",
    ]
    values = out["Mean_Uncertainty_Sigma"].tolist()
    colors = ["#333f9c"] + [
        "#1b5e63" if d >= 0 else "#b3491f" for d in out["Delta_vs_Full"].tolist()[1:]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=3)
    ax.axhline(full_sigma, color="#333f9c", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
    ax.set_ylabel("Mean cross-model disagreement  σ$_{model}$  (std. dev., 4 Tier 1 models)")
    ax.set_title(f"Tier 1 Agent Ablation — Mean Cross-Agent Disagreement (C₀, n={len(df):,})")
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(values) * 1.18)
    for bar, val, delta in zip(bars, values, out["Delta_vs_Full"].tolist()):
        label = f"{val:.4f}" if delta == 0 else f"{val:.4f}\n({delta:+.4f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.015,
                 label, ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    print(f"✅ Wrote {OUT_FIG}")
    print("\nNote: Tier 2 curation agents (Contradiction, Cross-Archive Linker, Grounded-OCR"
          " document extraction) and prototype auxiliary tooling (Temporal Historian, Bayesian"
          " Ranker, Frontier Auditor) are excluded by design -- per the Architectural Decoupling"
          " Rule, none of them participate in the core SAA/uncertainty calculation, so an"
          " uncertainty ablation over them would not measure anything real.")


if __name__ == "__main__":
    main()
