#!/usr/bin/env python3
"""
run_calibration_metrics.py
Implements Tier 1 Priority A: Uncertainty Calibration Framework.
Calculates ECE, Brier Score, AUROC, AUPRC and generates Reliability Diagrams.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[2]
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "results" / "multi_agent"

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Calculate bin counts to weight the ECE
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_total = np.bincount(binids, minlength=n_bins)
    non_empty_bins = bin_total > 0
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[non_empty_bins] / len(y_prob)))
    return float(ece), prob_true, prob_pred

def calculate_fpr95(y_true, y_prob):
    # Calculates the False Positive Rate at 95% True Positive Rate
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])

def main():
    if not SCORES_CSV.exists():
        print(f"❌ Error: {SCORES_CSV} not found.")
        return

    print("Loading agent comparison scores...")
    df = pd.read_csv(SCORES_CSV)
    
    # Define internal agents for calibration
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    df = df.dropna(subset=agent_cols)
    
    # 1. Compute Uncertainty Scores
    print("Calculating Unified Uncertainty Score (U)...")
    # σ_agents: standard deviation of agent scores
    df["sigma_agents"] = df[agent_cols].std(axis=1)
    # c_bar: mean confidence
    df["c_bar"] = df[agent_cols].mean(axis=1)
    # Unified U: alpha=0.6, beta=0.4 (omitting entropy H for now)
    df["unified_uncertainty_U"] = 0.6 * df["sigma_agents"] + 0.4 * (1 - df["c_bar"])
    
    # 2. Define Proxy Ground Truth for "Failure" (Ambiguous/Error state)
    # We define a failure as cases where the overall realism (c_bar) is below a strict threshold (0.4)
    # OR where the difference between maximum and minimum agent scores is > 0.6 (extreme conflict)
    df["max_diff"] = df[agent_cols].max(axis=1) - df[agent_cols].min(axis=1)
    df["proxy_failure_label"] = ((df["c_bar"] < 0.4) | (df["max_diff"] > 0.6)).astype(int)
    
    y_true = df["proxy_failure_label"].values
    if len(np.unique(y_true)) < 2:
        print("❌ Not enough variance in proxy labels to compute ROC AUC.")
        return

    # Raw YOLO confidence error predictor: 1 - existing_pipeline_agent
    y_raw_error_prob = 1.0 - df["existing_pipeline_agent"].values
    
    # Our Multi-Agent Uncertainty predictors
    y_disagreement_prob = df["sigma_agents"].values
    # Normalize disagreement to [0,1] for proper metric scaling
    y_disagreement_prob = y_disagreement_prob / (y_disagreement_prob.max() + 1e-9)
    
    y_unified_prob = df["unified_uncertainty_U"].values
    y_unified_prob = y_unified_prob / (y_unified_prob.max() + 1e-9)
    
    # 3. Experiment A2: Error Prediction Metrics
    print("\n--- Experiment A2: Error Prediction Metrics ---")
    metrics = {}
    predictors = {
        "Raw_YOLO_Confidence_Inverse": y_raw_error_prob,
        "Agent_Disagreement_Std": y_disagreement_prob,
        "Unified_Uncertainty_U": y_unified_prob
    }
    
    for name, y_prob in predictors.items():
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        fpr95 = calculate_fpr95(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        
        # Calculate ECE for failure prediction
        ece, prob_true, prob_pred = expected_calibration_error(y_true, y_prob)
        
        metrics[name] = {
            "AUROC": round(float(auroc), 4),
            "AUPRC": round(float(auprc), 4),
            "FPR95": round(float(fpr95), 4),
            "Brier_Score": round(float(brier), 4),
            "ECE": round(float(ece), 4)
        }
        print(f"[{name}] AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | FPR95: {fpr95:.4f} | ECE: {ece:.4f}")

    # 4. Generate Reliability Diagrams (Calibration Curves)
    print("\nGenerating Reliability Diagrams...")
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name, y_prob in predictors.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (ECE={metrics[name]['ECE']:.3f})")
        
    plt.xlabel("Mean Predicted Probability of Failure")
    plt.ylabel("Fraction of Actual Failures (Proxy)")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    cal_plot_path = OUTPUT_DIR / "calibration_reliability_diagram.png"
    plt.savefig(cal_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate Uncertainty Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(y_raw_error_prob, bins=50, alpha=0.5, label="Raw YOLO (1 - conf)", density=True)
    plt.hist(y_unified_prob, bins=50, alpha=0.5, label="Unified Uncertainty U", density=True)
    plt.title("Uncertainty Score Distribution")
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    
    hist_plot_path = OUTPUT_DIR / "uncertainty_histogram.png"
    plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics to JSON
    out_json = OUTPUT_DIR / "calibration_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Saved plots to {cal_plot_path} and {hist_plot_path}")
    print(f"✅ Saved metrics to {out_json}")

if __name__ == "__main__":
    main()
