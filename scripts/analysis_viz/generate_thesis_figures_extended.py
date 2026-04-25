#!/usr/bin/env python3
"""
Extended thesis figures — second set.

Produces:
  results/figures/thesis_scene_type_heatmap.png
  results/figures/thesis_agent_diversity_heatmap.png
  results/figures/thesis_hitl_simulation.png
  results/figures/thesis_ppn_distribution.png
  results/figures/thesis_calibration_curves.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
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

def save_fig(fig, path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ {path}")

# ── Scene-type heatmap ────────────────────────────────────────────────────────
def fig_scene_heatmap(scene_csv, out):
    if not scene_csv.exists():
        print(f"  ⚠ skip {scene_csv}"); return
    df = pd.read_csv(scene_csv)
    cols = ["existing_pipeline_agent_mean","agreement_agent_mean","scene_agent_mean",
            "vlm_agent_mean","monolithic_pipeline_agent_mean","comparison_fusion_score_mean"]
    col_labels = ["Object","Agreement","Scene","VLM","Monolithic","Fusion"]
    present = [c for c in cols if c in df.columns]
    if not present: return
    data   = df[present].to_numpy(dtype=float)
    scenes = df["scene_type"].tolist()

    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([col_labels[cols.index(c)] for c in present], fontsize=10)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels([f"{s.capitalize()}\n(n={int(df.iloc[i]['n_images']):,})" for i, s in enumerate(scenes)], fontsize=9)
    for i in range(len(scenes)):
        for j in range(len(present)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if data[i,j] > 0.5 else "white")
    plt.colorbar(im, ax=ax, label="Mean Score")
    ax.set_title("Agent Performance by Scene Type", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, out)

# ── Agent diversity correlation heatmap ──────────────────────────────────────
def fig_diversity_heatmap(div_csv, out):
    if not div_csv.exists():
        print(f"  ⚠ skip {div_csv}"); return
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
    save_fig(fig, out)

# ── HITL simulation multi-strategy ───────────────────────────────────────────
def fig_hitl_simulation(hitl_csv, out):
    if not hitl_csv.exists():
        print(f"  ⚠ skip {hitl_csv}"); return
    df = pd.read_csv(hitl_csv)
    strategies = df["strategy"].unique()
    colors = {"disagreement_driven":"#e15759","monolithic_confidence":"#f28e2b","random_sampling":"#4e79a7"}
    labels = {"disagreement_driven":"Coordinator (Disagreement)","monolithic_confidence":"Monolithic Confidence","random_sampling":"Random"}

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=STYLE["figure.facecolor"])
    for ax in axes: ax.set_facecolor(STYLE["axes.facecolor"])

    # Left: Precision@k
    for strat in strategies:
        sub = df[df["strategy"]==strat].sort_values("budget_ratio")
        axes[0].plot(sub["budget_ratio"]*100, sub["precision"], "o-",
                     color=colors.get(strat,"#aaa"), linewidth=2.5, markersize=7, label=labels.get(strat,strat))
    axes[0].set_xlabel("Review Budget (%)", fontsize=11); axes[0].set_ylabel("Precision (Issue Detection Rate)", fontsize=11)
    axes[0].set_title("HITL Triage: Precision across Budgets", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9, facecolor="#1a1a2e"); axes[0].grid(True); axes[0].set_ylim(0,1.1)

    # Right: Recall@k
    for strat in strategies:
        sub = df[df["strategy"]==strat].sort_values("budget_ratio")
        axes[1].plot(sub["budget_ratio"]*100, sub["recall"], "o-",
                     color=colors.get(strat,"#aaa"), linewidth=2.5, markersize=7, label=labels.get(strat,strat))
    axes[1].set_xlabel("Review Budget (%)", fontsize=11); axes[1].set_ylabel("Recall (Issues Captured)", fontsize=11)
    axes[1].set_title("HITL Triage: Recall across Budgets", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, facecolor="#1a1a2e"); axes[1].grid(True); axes[1].set_ylim(0,1.1)

    fig.tight_layout(pad=2)
    save_fig(fig, out)

# ── PPN size distribution ─────────────────────────────────────────────────────
def fig_ppn_dist(ppn_csv, out):
    if not ppn_csv.exists():
        print(f"  ⚠ skip {ppn_csv}"); return
    df = pd.read_csv(ppn_csv)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=STYLE["figure.facecolor"])
    for ax in axes: ax.set_facecolor(STYLE["axes.facecolor"])

    # Left: PPN size distribution
    sizes = df["n_images"].to_numpy()
    axes[0].hist(sizes, bins=40, color="#4e79a7", alpha=0.8, edgecolor="none")
    axes[0].axvline(np.median(sizes), color="#e15759", linestyle="--", linewidth=2, label=f"Median={int(np.median(sizes))}")
    axes[0].set_xlabel("Images per PPN Collection", fontsize=11); axes[0].set_ylabel("Number of PPNs", fontsize=11)
    axes[0].set_title("PPN Collection Size Distribution\n(1,722 Archival Publications)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9, facecolor="#1a1a2e"); axes[0].grid(True, axis="y")

    # Right: Fusion score vs PPN size scatter
    fus_col = "comparison_fusion_score_mean"
    if fus_col in df.columns:
        axes[1].scatter(df["n_images"], df[fus_col], alpha=0.3, s=8, color="#59a14f", rasterized=True)
        axes[1].set_xlabel("Images per PPN Collection", fontsize=11)
        axes[1].set_ylabel("Mean Fusion Score", fontsize=11)
        axes[1].set_title("Collection Size vs Mean Fusion Score\n(Cross-Collection Consistency)", fontsize=11, fontweight="bold")
        axes[1].grid(True)

    fig.tight_layout(pad=2)
    save_fig(fig, out)

# ── Calibration curves ────────────────────────────────────────────────────────
def fig_calibration(cal_csv, out):
    if not cal_csv.exists():
        print(f"  ⚠ skip {cal_csv}"); return
    df = pd.read_csv(cal_csv)
    predictors = df["predictor"].unique()
    colors_c = {"comparison_fusion_score":"#e15759","monolithic_pipeline_agent":"#b07aa1",
                "agreement_agent":"#f28e2b","vlm_agent":"#59a14f"}
    labels_c = {"comparison_fusion_score":"Coordinator Fusion","monolithic_pipeline_agent":"Monolithic",
                "agreement_agent":"Agreement Agent","vlm_agent":"VLM Agent"}

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    ax.plot([0,1],[0,1],"w--",alpha=0.4,linewidth=1.5,label="Perfect calibration")
    for pred in predictors:
        sub = df[df["predictor"]==pred].sort_values("confidence")
        ax.plot(sub["confidence"], sub["fraction_positive"], "o-",
                color=colors_c.get(pred,"#aaa"), linewidth=2.5, markersize=7, label=labels_c.get(pred,pred))
    ax.set_xlabel("Predicted Score (Confidence)", fontsize=11)
    ax.set_ylabel("Observed Quality Rate", fontsize=11)
    ax.set_title("Reliability Diagram: Calibration Curves\n(closer to diagonal = better calibrated)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1a1a2e"); ax.grid(True); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.15)
    save_fig(fig, out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--results-dir", default="results/multi_agent")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    base    = Path(args.base_dir).resolve()
    res_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else base / args.results_dir
    out_dir = Path(args.output_dir)  if Path(args.output_dir).is_absolute()  else base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Generating extended thesis figures...")
    fig_scene_heatmap(res_dir/"scene_type_performance.csv",   out_dir/"thesis_scene_type_heatmap.png")
    fig_diversity_heatmap(res_dir/"agent_diversity_matrix.csv", out_dir/"thesis_agent_diversity_heatmap.png")
    fig_hitl_simulation(res_dir/"hitl_simulation.csv",         out_dir/"thesis_hitl_simulation.png")
    fig_ppn_dist(res_dir/"ppn_analysis.csv",                   out_dir/"thesis_ppn_distribution.png")
    fig_calibration(res_dir/"calibration_analysis.csv",        out_dir/"thesis_calibration_curves.png")
    print(f"\n✅ All extended figures saved to {out_dir}")

if __name__ == "__main__":
    main()
