import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

# ECE implementation
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
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
    
    # We want to measure the prediction of presence of scene (gold_has_scene) using each agent's scores:
    # 1. existing_pipeline_agent (YOLO confidence)
    # 2. agreement_agent (SAA Disagreement - wait! agreement_agent is 1.0 - agreement? Let's check values or see if agreement_agent aligns)
    # Let's print summary stats for agents:
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    
    for col in agent_cols:
        y_prob = joined[col].values
        auc = roc_auc_score(y_true, y_prob)
        # Check if inverse is better (in case it is inverted calibration/uncertainty)
        auc_inv = roc_auc_score(y_true, 1.0 - y_prob)
        best_auc = max(auc, auc_inv)
        best_prob = y_prob if auc >= auc_inv else 1.0 - y_prob
        ece = expected_calibration_error(y_true, best_prob)
        print(f"Agent: {col}")
        print(f"  ROC-AUC: {auc:.4f} (Inv: {auc_inv:.4f}) -> Best: {best_auc:.4f}")
        print(f"  ECE (on best): {ece:.4f}")
        
    # Let's also evaluate the AWLF model (trained LogisticRegression)
    X = joined[agent_cols].copy()
    y = joined["gold_has_scene"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    awlf_auc = roc_auc_score(y_test, test_probs)
    awlf_ece = expected_calibration_error(y_test.values, test_probs)
    print("AWLF (Logistic-Regression Meta-Classifier) on Test Split:")
    print(f"  ROC-AUC: {awlf_auc:.4f}")
    print(f"  ECE: {awlf_ece:.4f}")

if __name__ == "__main__":
    main()
