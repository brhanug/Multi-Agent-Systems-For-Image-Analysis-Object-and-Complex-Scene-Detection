#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path
import json

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "results/multi_agent"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1]
    p = p.rsplit(".", 1)[0]
    parts = p.split("_")
    if len(parts) >= 2 and parts[0].startswith("PPN"):
        return "_".join(parts[-2:])
    return p

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    non_empty_bins = bin_total > 0
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[non_empty_bins] / len(y_prob)))
    return float(ece)

def main():
    print("🚀 Running PATH 1 REAL EXPERIMENT: Disagreement vs. Single-Model Confidence on 801 Expert Images...")
    
    # 1. Load expert gold labels and restrict to reviewed set (n=801)
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
    print(f"  Matched exactly {len(joined)} reviewed images with agent scores.")
    
    # 4. Define prediction errors (Classification failure against human expert)
    # A failure/error occurs when the model predicts presence (score >= 0.5) but human says absence (gold_has_scene = 0)
    # OR model predicts absence (score < 0.5) but human says presence (gold_has_scene = 1).
    gold_label = joined["gold_has_scene"].values
    
    # We analyze prediction errors of the Monolithic Baseline
    joined["mono_error"] = (joined["monolithic_pipeline_agent"] >= 0.5).astype(int) != gold_label
    mono_errors = joined["mono_error"].astype(int).values
    
    # We analyze prediction errors of the Object Agent (YOLO)
    joined["yolo_error"] = (joined["existing_pipeline_agent"] >= 0.5).astype(int) != gold_label
    yolo_errors = joined["yolo_error"].astype(int).values
    
    print(f"  Monolithic Baseline Errors: {mono_errors.sum()} / {len(joined)} ({mono_errors.mean()*100:.2f}%)")
    print(f"  Object Agent (YOLO) Errors: {yolo_errors.sum()} / {len(joined)} ({yolo_errors.mean()*100:.2f}%)")
    
    # 5. Define Predictors of Error
    # Predictor A: Inverse of single-model detector confidence (1 - existing_pipeline_agent)
    yolo_conf_inv = 1.0 - joined["existing_pipeline_agent"].values
    
    # Predictor B: Inverse of monolithic pipeline confidence (1 - monolithic_pipeline_agent)
    mono_conf_inv = 1.0 - joined["monolithic_pipeline_agent"].values
    
    # Predictor C: Inter-agent disagreement (standard deviation of active validation agents)
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    joined["sigma_agents"] = joined[agent_cols].std(axis=1)
    disagreement_std = joined["sigma_agents"].values
    # Normalize to [0,1]
    disagreement_std = disagreement_std / (disagreement_std.max() + 1e-9)
    
    # Predictor D: Unified Uncertainty Score (U)
    joined["c_bar"] = joined[agent_cols].mean(axis=1)
    joined["unified_U"] = 0.6 * joined["sigma_agents"] + 0.4 * (1.0 - joined["c_bar"])
    unified_U = joined["unified_U"].values
    unified_U = unified_U / (unified_U.max() + 1e-9)
    
    # 6. Evaluate Error Predictors for Monolithic Errors
    print("\n📊 MONOLITHIC PREDICTION ERROR DETECTORS:")
    monolithic_metrics = {}
    predictors_mono = {
        "Raw Monolithic Confidence Inverse": mono_conf_inv,
        "Inter-Agent Disagreement (Std)": disagreement_std,
        "Unified Uncertainty Score (U)": unified_U
    }
    for name, scores in predictors_mono.items():
        auc_roc = roc_auc_score(mono_errors, scores)
        auc_pr = average_precision_score(mono_errors, scores)
        brier = brier_score_loss(mono_errors, scores)
        ece = expected_calibration_error(mono_errors, scores)
        monolithic_metrics[name] = {
            "AUROC": round(float(auc_roc), 4),
            "AUPRC": round(float(auc_pr), 4),
            "Brier": round(float(brier), 4),
            "ECE": round(float(ece), 4)
        }
        print(f"  [{name:35s}] AUROC: {auc_roc:.4f} | AUPRC: {auc_pr:.4f} | ECE: {ece:.4f}")
        
    # 7. Evaluate Error Predictors for YOLO Errors
    print("\n📊 OBJECT DETECTOR (YOLO) PREDICTION ERROR DETECTORS:")
    yolo_metrics = {}
    predictors_yolo = {
        "Raw YOLO Confidence Inverse": yolo_conf_inv,
        "Inter-Agent Disagreement (Std)": disagreement_std,
        "Unified Uncertainty Score (U)": unified_U
    }
    for name, scores in predictors_yolo.items():
        auc_roc = roc_auc_score(yolo_errors, scores)
        auc_pr = average_precision_score(yolo_errors, scores)
        brier = brier_score_loss(yolo_errors, scores)
        ece = expected_calibration_error(yolo_errors, scores)
        yolo_metrics[name] = {
            "AUROC": round(float(auc_roc), 4),
            "AUPRC": round(float(auc_pr), 4),
            "Brier": round(float(brier), 4),
            "ECE": round(float(ece), 4)
        }
        print(f"  [{name:35s}] AUROC: {auc_roc:.4f} | AUPRC: {auc_pr:.4f} | ECE: {ece:.4f}")
        
    # 8. Generate Reliability Diagrams for Monolithic Errors
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, scores in predictors_mono.items():
        prob_true, prob_pred = calibration_curve(mono_errors, scores, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (ECE={monolithic_metrics[name]['ECE']:.3f})")
    plt.xlabel("Mean Predicted Probability of Error")
    plt.ylabel("Fraction of Actual Classification Errors")
    plt.title("Reliability Diagram: Predicting Monolithic Baseline Classification Errors (n=801)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "real_expert_calibration_reliability_mono.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Generate Reliability Diagrams for YOLO Errors
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, scores in predictors_yolo.items():
        prob_true, prob_pred = calibration_curve(yolo_errors, scores, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (ECE={yolo_metrics[name]['ECE']:.3f})")
    plt.xlabel("Mean Predicted Probability of Error")
    plt.ylabel("Fraction of Actual Classification Errors")
    plt.title("Reliability Diagram: Predicting Object Agent (YOLO) Classification Errors (n=801)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "real_expert_calibration_reliability_yolo.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Save results to JSON
    out_json = OUTPUT_DIR / "real_expert_rq2_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "monolithic_prediction_error_detectors": monolithic_metrics,
            "yolo_prediction_error_detectors": yolo_metrics
        }, f, indent=4)
    print(f"\n✅ Saved all metrics and reliability diagrams to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
