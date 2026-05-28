import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from pathlib import Path
import os

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR = BASE / "latex_assets"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
    print("Running Expert Calibration Analysis...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load Data
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"Matched {len(joined)} expert images.")
    
    y_true = joined["gold_has_scene"].values
    
    # Probabilities
    yolo_probs = joined["existing_pipeline_agent"].values
    vlm_probs = joined["vlm_agent"].values
    
    # AWLF MAS Probabilities
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    X = joined[agent_cols].copy()
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X, y_true)
    mas_probs = clf.predict_proba(X)[:, 1]
    
    # Plotting setup
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    colors = {'YOLOv11': 'red', 'VLM': 'blue', 'MAS (AWLF)': 'green'}
    probs_dict = {'YOLOv11': yolo_probs, 'VLM': vlm_probs, 'MAS (AWLF)': mas_probs}
    
    for name, probs in probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_true, probs)
        
        ax1.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier: {brier:.3f})", color=colors[name])
        ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2, color=colors[name])
        
        print(f"{name} Brier Score: {brier:.4f}")
    
    ax1.set_ylabel("Fraction of Positives (True Label)")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Reliability Diagram (Calibration Curve) on 801 Expert Annotations")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Image Count")
    ax2.legend(loc="upper center", ncol=3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = OUT_DIR / "expert_calibration_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Reliability Diagram to {out_path}")

if __name__ == "__main__":
    main()
