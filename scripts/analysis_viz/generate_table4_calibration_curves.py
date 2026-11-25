#!/usr/bin/env python3
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, STORED_COLUMN_THRESHOLD  # deliberate 0.55; see saa_decision.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_PATH = BASE / "latex_assets" / "uncertainty_calibration_curve.png"

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

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

def calibration_curve_custom(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    prob_true = []
    prob_pred = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
        if np.sum(in_bin) > 0:
            prob_true.append(np.mean(y_true[in_bin]))
            prob_pred.append(np.mean(y_prob[in_bin]))
    return np.array(prob_true), np.array(prob_pred)

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
    apply_style()
    
    # 1. Load Data
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    
    # 2. Target variable: Coordinator Error
    joined["is_error"] = (joined["comparison_fusion_score"] >= STORED_COLUMN_THRESHOLD).astype(int) != joined["gold_has_scene"]
    y_err = joined["is_error"].astype(int).values
    
    # 3. Predictors of error
    # SAA Disagreement
    agent_cols_4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    u_saa = joined[agent_cols_4].std(axis=1).values
    
    # Temp-Scaled Entropy
    p_sce = joined["scene_agent"].values
    entropy = - (p_sce * np.log2(p_sce + 1e-12) + (1-p_sce) * np.log2(1-p_sce + 1e-12))
    temp_entropy = entropy / 1.5
    
    # MC Dropout
    mc_dropout = 1.0 - joined["existing_pipeline_agent"].values
    
    # Deep Ensemble
    agent_cols_6 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent", "restoration_agent", "document_agent"]
    deep_ensemble = joined[agent_cols_6].std(axis=1).values
    
    methods = [
        (u_saa, "Multi-Agent SAA (Ours)", "#ff7f0e"),
        (temp_entropy, "Temperature-Scaled Entropy", "#2ca02c"),
        (mc_dropout, "Monte Carlo Dropout", "#1f77b4"),
        (deep_ensemble, "Deep Ensemble Uncertainty", "#d62728")
    ]
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, linewidth=1.5, label="Perfect calibration")
    
    for u, label, color in methods:
        u_norm = u / np.max(u) if np.max(u) > 0 else u
        # Clip to ensure no floating point overflow/underflow violates [0, 1] range
        u_norm = np.clip(u_norm, 0.0, 1.0)
        prob_true, prob_pred = calibration_curve_custom(y_err, u_norm, n_bins=10)
        ece = expected_calibration_error(y_err, u_norm)
        ax.plot(prob_pred, prob_true, "s-", linewidth=2.5, markersize=7, label=f"{label} (ECE = {ece:.4f})", color=color)
        
    ax.set_xlabel("Mean Predicted Probability of Classification Error", fontsize=11)
    ax.set_ylabel("Fraction of Actual Classification Errors", fontsize=11)
    ax.set_title("Reliability Diagram: Error Calibration on Human-Verified Baseline (n=801)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1a1a2e")
    ax.grid(True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated and saved calibration curves to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
