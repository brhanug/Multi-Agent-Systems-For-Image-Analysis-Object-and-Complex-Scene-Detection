#!/usr/bin/env python3
"""
analyze_latency_vs_calibration.py
-----------------------------------
Task: Cost-Latency Pareto Frontier Analysis

Models the trade-offs between system latency (computational cost) and
uncertainty/calibration performance (Uncertainty ROC-AUC, ECE, and Error Recall @ 20% Budget)
across different system configurations.

Identifies the Pareto-efficient configurations (where you cannot improve performance
without increasing computational latency).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Compute latency estimates (in seconds per image, based on standard RTX 4090 runtime)
# YOLOv11m-seg: 0.05s
# SigLIP (Scene): 0.02s
# LLaVA-OneVision (VLM): 3.00s
# SAA Agreement: 0.05s (lightweight bounding box / text parsing)

CONFIGS = [
    {
        "name": "Scene only",
        "latency": 0.02,
        "roc_auc": 0.5100,
        "ece": 0.0655,
        "recall_20": 0.0008,
        "hitl_prec": 0.0025,
        "color": "#10B981"  # Emerald
    },
    {
        "name": "YOLO only",
        "latency": 0.05,
        "roc_auc": 0.6106,
        "ece": 0.0813,
        "recall_20": 0.2062,
        "hitl_prec": 0.5125,
        "color": "#3B82F6"  # Blue
    },
    {
        "name": "YOLO + Scene",
        "latency": 0.07,
        "roc_auc": 0.5334,
        "ece": 0.0996,
        "recall_20": 0.1929,
        "hitl_prec": 0.4900,
        "color": "#8B5CF6"  # Purple
    },
    {
        "name": "VLM only",
        "latency": 3.00,
        "roc_auc": 0.5629,
        "ece": 0.0525,
        "recall_20": 0.0943,
        "hitl_prec": 0.2625,
        "color": "#F59E0B"  # Amber
    },
    {
        "name": "VLM + Scene",
        "latency": 3.02,
        "roc_auc": 0.5159,
        "ece": 0.1689,
        "recall_20": 0.0870,
        "hitl_prec": 0.2450,
        "color": "#EC4899"  # Pink
    },
    {
        "name": "YOLO + VLM",
        "latency": 3.05,
        "roc_auc": 0.5309,
        "ece": 0.1723,
        "recall_20": 0.1885,
        "hitl_prec": 0.5250,
        "color": "#EF4444"  # Red
    },
    {
        "name": "YOLO + VLM + Scene",
        "latency": 3.07,
        "roc_auc": 0.5183,
        "ece": 0.1862,
        "recall_20": 0.1670,
        "hitl_prec": 0.4650,
        "color": "#6B7280"  # Gray
    },
    {
        "name": "Full Agent 0 (Ours)",
        "latency": 3.12,
        "roc_auc": 0.5423,
        "ece": 0.1584,
        "recall_20": 0.1279,
        "hitl_prec": 0.3600,
        "color": "#4F46E5"  # Indigo (proposed core)
    }
]

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the Pareto efficient points.
    costs: An (n_points, n_costs) array where smaller is better.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point that is strictly better in at least one dimension
            # or equal in all, and not dominated by any other point
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient

def main():
    print("📈 Running Latency vs. Performance Pareto Frontier Analysis...")
    df = pd.DataFrame(CONFIGS)
    
    # Define multi-objective optimization targets:
    # 1. Minimize Latency (x-axis)
    # 2. Maximize HITL Precision (y-axis target 1)
    # 3. Minimize ECE (y-axis target 2)
    # 4. Maximize Error Recall (y-axis target 3)
    
    # We evaluate Pareto efficiency on (Latency, 1 - HITL_Precision)
    # and (Latency, ECE)
    
    costs_prec = np.column_stack((df["latency"].values, 1.0 - df["hitl_prec"].values))
    df["pareto_precision"] = is_pareto_efficient(costs_prec)
    
    costs_ece = np.column_stack((df["latency"].values, df["ece"].values))
    df["pareto_ece"] = is_pareto_efficient(costs_ece)
    
    print("\n" + "="*85)
    print(f"  {'Configuration':<26} {'Latency (s)':<12} {'ECE':<8} {'HITL-P':<8} {'Rec@20%':<8} {'Pareto (P/ECE)':<15}")
    print("="*85)
    for _, row in df.sort_values("latency").iterrows():
        p_status = f"{'Yes' if row['pareto_precision'] else 'No'}/{'Yes' if row['pareto_ece'] else 'No'}"
        print(f"  {row['name']:<26} {row['latency']:<12.3f} {row['ece']:<8.4f} {row['hitl_prec']:<8.4f} {row['recall_20']:<8.4f} {p_status:<15}")
    print("="*85)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#0f0f1a')
    
    BG = '#1a1a2e'
    GRID = '#2a2a4a'
    TEXT = '#e0e0e0'
    
    for ax in axes:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(color=GRID, linestyle='--', alpha=0.5)

    # Plot 1: Latency vs. HITL Triage Precision
    axes[0].set_title("Pareto Frontier: Latency vs. HITL Precision\n(Maximize Precision, Minimize Latency)", color=TEXT, fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel("Operational Latency (seconds per image)")
    axes[0].set_ylabel("HITL Triage Precision @ 20% Budget")
    axes[0].set_xscale("log")
    
    # Plot all points
    for _, row in df.iterrows():
        axes[0].scatter(row["latency"], row["hitl_prec"], color=row["color"], s=150, zorder=3, edgecolors='black', label=row["name"])
        axes[0].text(row["latency"] * 1.1, row["hitl_prec"] - 0.015, row["name"], color=TEXT, fontsize=8, fontweight='semibold')
        
    # Draw Pareto Line
    efficient_df = df[df["pareto_precision"]].sort_values("latency")
    axes[0].plot(efficient_df["latency"].values, efficient_df["hitl_prec"].values, color="#FF2E93", linestyle="--", linewidth=1.5, zorder=2, label="Pareto Frontier")
    
    # Plot 2: Latency vs. ECE
    axes[1].set_title("Pareto Frontier: Latency vs. Calibration ECE\n(Minimize ECE, Minimize Latency)", color=TEXT, fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("Operational Latency (seconds per image)")
    axes[1].set_ylabel("Expected Calibration Error (ECE)")
    axes[1].set_xscale("log")
    
    # Plot all points
    for _, row in df.iterrows():
        axes[1].scatter(row["latency"], row["ece"], color=row["color"], s=150, zorder=3, edgecolors='black')
        axes[1].text(row["latency"] * 1.1, row["ece"] + 0.003, row["name"], color=TEXT, fontsize=8, fontweight='semibold')
        
    # Draw Pareto Line
    efficient_ece_df = df[df["pareto_ece"]].sort_values("latency")
    axes[1].plot(efficient_ece_df["latency"].values, efficient_ece_df["ece"].values, color="#FF2E93", linestyle="--", linewidth=1.5, zorder=2, label="Pareto Frontier")

    plt.tight_layout()
    
    # Save Paths
    base_dir = Path("/data/brhanu/thesis_project")
    out_path = base_dir / "latex_assets" / "latency_pareto_frontier.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎨 Pareto Frontier visualization successfully saved to: {out_path}")
    print("\n💡 INTERPRETATION OF RESULTS:")
    print("  1. 'YOLO only' dominates 'YOLO + Scene' and 'VLM + Scene' due to high standalone speed and good precision.")
    print("  2. 'YOLO + VLM' is highly Pareto-efficient for precision (HITL-P = 0.5250) but takes 3.05s.")
    print("  3. 'VLM only' is Pareto-efficient for calibration ECE (ECE = 0.0525) but exhibits lower triage recall.")
    print("  4. The analysis shows that for high-throughput pipelines, the best trade-off is 'YOLO only' (0.05s) if triage precision is enough, "
          "and 'YOLO + VLM' (3.05s) for high-accuracy semantic check, which justifies the Agent 3 pre-screening bypass ranker.")

if __name__ == "__main__":
    main()
