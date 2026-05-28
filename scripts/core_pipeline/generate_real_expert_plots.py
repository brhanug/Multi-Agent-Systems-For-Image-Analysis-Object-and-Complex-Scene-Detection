#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
LATEX_ASSETS_DIR = BASE / "latex_assets"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    non_empty_bins = bin_total > 0
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[non_empty_bins] / len(y_prob)))
    return float(ece)

def main():
    print("🎨 Generating REAL EXPERT calibration and ROC plots...")
    
    # 1. Load expert gold labels (n=801 reviewed)
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    # 2. Load agent predictions
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    # 3. Join
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"  Matched {len(joined)} expert reviewed images.")
    
    # 4. Target variable: Model classification error (YOLO error)
    gold_label = joined["gold_has_scene"].values
    joined["yolo_error"] = (joined["existing_pipeline_agent"] >= 0.5).astype(int) != gold_label
    y_true = joined["yolo_error"].astype(int).values
    
    # 5. Predictors of error
    # Predictor 1: Raw YOLO Confidence Inverse
    yolo_conf_inv = 1.0 - joined["existing_pipeline_agent"].values
    
    # Predictor 2: Inter-Agent Disagreement (Std)
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    joined["sigma_agents"] = joined[agent_cols].std(axis=1)
    disagreement_std = joined["sigma_agents"].values
    disagreement_std = disagreement_std / (disagreement_std.max() + 1e-9)
    
    # Predictor 3: Unified Uncertainty Score (U)
    joined["c_bar"] = joined[agent_cols].mean(axis=1)
    joined["unified_U"] = 0.6 * joined["sigma_agents"] + 0.4 * (1.0 - joined["c_bar"])
    unified_U = joined["unified_U"].values
    unified_U = unified_U / (unified_U.max() + 1e-9)
    
    # --- PLOT 1: ROC CURVES (thesis_rq2_roc.png) ---
    print("  Creating ROC curve: thesis_rq2_roc.png...")
    plt.figure(figsize=(8, 6))
    
    for scores, label, color in [
        (yolo_conf_inv, "Raw YOLO Confidence Inverse", "#1f77b4"),
        (disagreement_std, "Inter-Agent Disagreement (Std)", "#ff7f0e"),
        (unified_U, "Unified Uncertainty U", "#2ca02c")
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.4f})", color=color)
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title('ROC Curve: Error Prediction on Real Expert Baseline (n=801)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_plot_path = LATEX_ASSETS_DIR / "thesis_rq2_roc.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curve to {roc_plot_path}")
    
    # --- PLOT 2: CALIBRATION CURVES (thesis_calibration_curves.png) ---
    print("  Creating Calibration Curve: thesis_calibration_curves.png...")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for scores, label, color in [
        (yolo_conf_inv, "Raw YOLO Confidence Inverse", "#1f77b4"),
        (disagreement_std, "Inter-Agent Disagreement (Std)", "#ff7f0e"),
        (unified_U, "Unified Uncertainty U", "#2ca02c")
    ]:
        prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy='uniform')
        ece = expected_calibration_error(y_true, scores)
        plt.plot(prob_pred, prob_true, "s-", label=f"{label} (ECE = {ece:.4f})", color=color)
        
    plt.xlabel("Mean Predicted Probability of Classification Error", fontsize=11)
    plt.ylabel("Fraction of Actual Classification Errors", fontsize=11)
    plt.title("Reliability Diagram: Error Calibration on Real Expert Baseline (n=801)", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    cal_plot_path = LATEX_ASSETS_DIR / "thesis_calibration_curves.png"
    plt.savefig(cal_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Calibration Curve to {cal_plot_path}")
    
    print("🏁 Done!")

if __name__ == "__main__":
    main()
