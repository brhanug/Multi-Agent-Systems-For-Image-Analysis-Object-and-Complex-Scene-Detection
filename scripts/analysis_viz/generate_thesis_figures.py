#!/usr/bin/env python3
"""
Generate publication-ready thesis figures from multi-agent evaluation results.

Produces:
  results/figures/thesis_agent_distributions.png  - violin/box plots of agent scores
  results/figures/thesis_mono_vs_fusion.png       - scatter plot monolithic vs fusion
  results/figures/thesis_rq2_roc.png              - ROC curves for disagreement predictors
  results/figures/thesis_complexity_bars.png      - complexity stratification bar chart
  results/figures/thesis_ablation_impact.png      - ablation delta bar chart
  results/figures/thesis_hitl_efficiency.png      - HITL precision@k vs random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Use Agg backend (no display needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


PALETTE = {
    "existing_pipeline_agent": "#4e79a7",
    "agreement_agent":         "#f28e2b",
    "scene_agent":             "#59a14f",
    "vlm_agent":               "#e15759",
    "restoration_agent":       "#76b7b2",
    "document_agent":          "#edc948",
    "monolithic_pipeline_agent": "#b07aa1",
    "comparison_fusion_score":   "#ff9da7",
}

AGENT_LABELS = {
    "existing_pipeline_agent": "Object\n(YOLO)",
    "agreement_agent":         "Agreement\n(SAA)",
    "scene_agent":             "Scene\n(CLIP)",
    "vlm_agent":               "VLM\n(LLaVA)",
    "restoration_agent":       "Restoration\n(SRS)",
    "document_agent":          "Document\n(Kosmos)",
    "monolithic_pipeline_agent": "Monolithic\nBaseline",
    "comparison_fusion_score":   "Coordinator\nFusion",
}

STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#0f3460",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
}


def apply_style() -> None:
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {path}")


def fig_agent_distributions(df: pd.DataFrame, out: Path) -> None:
    apply_style()
    cols = list(AGENT_LABELS.keys())
    data = [df[c].dropna().to_numpy() for c in cols if c in df.columns]
    labels = [AGENT_LABELS[c] for c in cols if c in df.columns]
    colors = [PALETTE[c] for c in cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=False)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("#ffffff")
    parts["cmedians"].set_linewidth(2)

    # Overlay box plots
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.2,
                    patch_artist=True, showcaps=False, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color("#666666")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Agent Score (normalized)", fontsize=11)
    ax.set_title("Agent Score Distributions across 12,110 Images", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y")
    ax.axvline(x=5.5, color="#555555", linestyle="--", alpha=0.6)
    ax.text(5.7, 0.97, "Baseline →", fontsize=8, color="#aaaaaa")

    save_fig(fig, out)


def fig_mono_vs_fusion(df: pd.DataFrame, out: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])

    mono = df["monolithic_pipeline_agent"].to_numpy()
    fus  = df["comparison_fusion_score"].to_numpy()

    # Density-coloured scatter
    density = np.zeros(len(mono))
    for i in range(len(mono)):
        mask = (np.abs(mono - mono[i]) < 0.05) & (np.abs(fus - fus[i]) < 0.05)
        density[i] = mask.sum()

    sc = ax.scatter(mono, fus, c=density, cmap="plasma", alpha=0.3, s=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Local density")

    # Diagonal reference (mono == fusion)
    ax.plot([0, 1], [0, 1], "w--", alpha=0.5, linewidth=1, label="mono = fusion")

    # Mean markers
    ax.axvline(x=float(np.mean(mono)), color="#b07aa1", linestyle=":", linewidth=2, label=f"Mono mean={np.mean(mono):.3f}")
    ax.axhline(y=float(np.mean(fus)),  color="#ff9da7", linestyle=":", linewidth=2, label=f"Fusion mean={np.mean(fus):.3f}")

    ax.set_xlabel("Monolithic Pipeline Agent Score", fontsize=11)
    ax.set_ylabel("Coordinator Fusion Score", fontsize=11)
    ax.set_title("Monolithic vs Coordinator Fusion: Per-Image Scores\n(12,110 images)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#0f3460")
    ax.grid(True)

    save_fig(fig, out)


def fig_rq2_roc(roc_csv: Path, out: Path) -> None:
    if not roc_csv.exists():
        print(f"  ⚠️  Skipping ROC figure (missing {roc_csv})")
        return

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])

    roc_df = pd.read_csv(roc_csv)
    colors_roc = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b"]
    for i, (name, grp) in enumerate(roc_df.groupby("predictor")):
        grp_sorted = grp.sort_values("fpr")
        ax.plot(grp_sorted["fpr"], grp_sorted["tpr"],
                color=colors_roc[i % len(colors_roc)],
                linewidth=2,
                label=name.replace("_", " ").title())

    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("RQ2: ROC Curves — Issue Prediction by Uncertainty Signal", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#0f3460")
    ax.grid(True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_fig(fig, out)


def fig_complexity_bars(complexity_csv: Path, out: Path) -> None:
    if not complexity_csv.exists():
        print(f"  ⚠️  Skipping complexity figure (missing {complexity_csv})")
        return

    apply_style()
    df = pd.read_csv(complexity_csv)

    bins = df["complexity_bin"].tolist()
    x = np.arange(len(bins))
    width = 0.25

    mono_col = "monolithic_pipeline_agent_mean"
    fus_col  = "comparison_fusion_score_mean"
    vlm_col  = "vlm_agent_mean"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=STYLE["figure.facecolor"])

    # Left: mean scores by bin
    ax = axes[0]
    ax.set_facecolor(STYLE["axes.facecolor"])
    if mono_col in df.columns:
        ax.bar(x - width, df[mono_col], width, label="Monolithic", color="#b07aa1", alpha=0.85)
    if fus_col in df.columns:
        ax.bar(x,         df[fus_col],  width, label="Fusion",     color="#ff9da7", alpha=0.85)
    if vlm_col in df.columns:
        ax.bar(x + width, df[vlm_col],  width, label="VLM Agent",  color="#e15759", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=20)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Agent Scores by Scene Complexity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1a1a2e")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y")

    # Right: HITL issue rate by bin
    ax2 = axes[1]
    ax2.set_facecolor(STYLE["axes.facecolor"])
    if "proxy_issue_rate" in df.columns:
        colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bins)))  # type: ignore
        bars = ax2.bar(x, df["proxy_issue_rate"], width=0.5, color=colors_bar, alpha=0.9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(bins, rotation=20)
        ax2.set_ylabel("Proxy Issue Rate", fontsize=11)
        ax2.set_title("Issue Rate by Scene Complexity", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, axis="y")

    fig.tight_layout(pad=2)
    save_fig(fig, out)


def fig_ablation(ablation_csv: Path, out: Path) -> None:
    if not ablation_csv.exists():
        print(f"  ⚠️  Skipping ablation figure (missing {ablation_csv})")
        return

    apply_style()
    df = pd.read_csv(ablation_csv).sort_values("delta_vs_full", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])

    deltas = df["delta_vs_full"].to_numpy()
    labels = [r["ablation"].replace("without_", "w/o ").replace("_", " ") for _, r in df.iterrows()]
    colors_ab = ["#e15759" if d < 0 else "#59a14f" for d in deltas]

    ax.barh(range(len(deltas)), deltas, color=colors_ab, alpha=0.85, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(x=0, color="white", linewidth=1, linestyle="--")
    ax.set_xlabel("Δ Fusion Mean (vs Full System)", fontsize=11)
    ax.set_title("Ablation Study: Leave-One-Agent-Out Impact on Fusion Score", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x")

    for i, (d, l) in enumerate(zip(deltas, labels)):
        ax.text(d + (0.001 if d >= 0 else -0.001), i,
                f"{d:+.4f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color="white")

    save_fig(fig, out)


def fig_hitl_efficiency(hitl_csv: Path, out: Path) -> None:
    if not hitl_csv.exists():
        print(f"  ⚠️  Skipping HITL figure (missing {hitl_csv})")
        return

    apply_style()
    df = pd.read_csv(hitl_csv)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])

    x = df["top_k_ratio"].to_numpy() * 100
    prec = df["precision_at_k"].to_numpy()
    rand = df["random_precision"].to_numpy()

    ax.plot(x, prec, "o-", color="#e15759", linewidth=2.5, markersize=8, label="Disagreement-Based Triage")
    ax.plot(x, rand, "s--", color="#4e79a7", linewidth=2, markersize=7, label="Random Baseline", alpha=0.8)

    ax.fill_between(x, rand, prec, alpha=0.15, color="#e15759", label="Efficiency Gain")

    ax.set_xlabel("Top-k Review Budget (%)", fontsize=11)
    ax.set_ylabel("Precision (Issue Detection Rate)", fontsize=11)
    ax.set_title("RQ4: HITL Triage Efficiency\nDisagreement-Based vs Random Sampling", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#0f3460")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(5, 35)
    ax.grid(True)

    if len(df) > 0:
        lift = df["lift_vs_random"].iloc[0]
        ax.text(0.05, 0.15, f"Lift vs Random: {lift:.2f}×",
                transform=ax.transAxes, fontsize=12, color="#e15759", fontweight="bold")

    save_fig(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis figures.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--results-dir", default="results/multi_agent")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    base    = Path(args.base_dir).resolve()
    res_dir = Path(args.results_dir)
    if not res_dir.is_absolute():
        res_dir = base / res_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Generating thesis figures...")

    # Load main scores
    scores_path = res_dir / "agent_comparison_scores.csv"
    if not scores_path.exists():
        print(f"❌ Missing {scores_path} — run run_agent_comparison.py first")
        return
    df = pd.read_csv(scores_path)

    fig_agent_distributions(df, out_dir / "thesis_agent_distributions.png")
    fig_mono_vs_fusion(df, out_dir / "thesis_mono_vs_fusion.png")
    fig_rq2_roc(res_dir / "rq2_roc_curve.csv", out_dir / "thesis_rq2_roc.png")
    fig_complexity_bars(res_dir / "complexity_deep_analysis.csv", out_dir / "thesis_complexity_bars.png")
    fig_ablation(res_dir / "research_ablation_summary.csv", out_dir / "thesis_ablation_impact.png")
    fig_hitl_efficiency(res_dir / "research_hitl_efficiency.csv", out_dir / "thesis_hitl_efficiency.png")

    print(f"\n✅ All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
