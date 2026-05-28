#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import json

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "results/multi_agent"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
    print("🚀 Running PATH 3 REAL EXPERIMENT: Training AWLF on Real Human Ground-Truth labels...")
    
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
    print(f"  Matched exactly {len(joined)} reviewed images with agent scores.")
    
    # 4. Features & Target
    feature_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    X = joined[feature_cols].copy()
    y = joined["gold_has_scene"].copy()
    
    # 5. Split into 60% Train (n=480) and 40% Test (n=321) with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    print(f"  Split data into {len(X_train)} Train samples and {len(X_test)} Test samples.")
    
    # 6. Train the AWLF Discriminator on Real Human Labels
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get learned weights
    weights = dict(zip(feature_cols, clf.coef_[0].tolist()))
    print("\n📦 LEARNED AWLF WEIGHTS (TRAINED ON REAL HUMAN LABELS):")
    for k, v in weights.items():
        print(f"  {k:30s} : {v:+.4f}")
    print(f"  Intercept                      : {clf.intercept_[0]:+.4f}")
    
    # 7. Evaluate on held-out Test Split
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    
    awlf_acc = accuracy_score(y_test, test_preds)
    awlf_prec = precision_score(y_test, test_preds)
    awlf_rec = recall_score(y_test, test_preds)
    awlf_f1 = f1_score(y_test, test_preds)
    awlf_auc = roc_auc_score(y_test, test_probs)
    
    # 8. Evaluate Baselines on the exact same Test Split
    baselines = {
        "Heuristic Object Agent (YOLO)": X_test["existing_pipeline_agent"],
        "Heuristic Agreement Agent": X_test["agreement_agent"],
        "Heuristic VLM Agent": X_test["vlm_agent"],
        "Monolithic Pipeline Baseline": joined.loc[X_test.index, "monolithic_pipeline_agent"]
    }
    
    results = []
    results.append({
        "Model / Strategy": "Trained AWLF Discriminator (Ours)",
        "Precision": awlf_prec,
        "Recall": awlf_rec,
        "F1": awlf_f1,
        "Accuracy": awlf_acc,
        "ROC-AUC": awlf_auc
    })
    
    for name, scores in baselines.items():
        preds = (scores >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, scores)
        results.append({
            "Model / Strategy": name,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Accuracy": acc,
            "ROC-AUC": auc
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n📈 E3/E4 HELD-OUT TEST PERFORMANCE COMPARISON (n=321 Expert Images):")
    print(df_results.to_string(index=False))
    
    # 9. Save outputs
    report_path = OUTPUT_DIR / "real_expert_awlf_results.json"
    with open(report_path, "w") as f:
        json.dump({
            "learned_weights": weights,
            "intercept": float(clf.intercept_[0]),
            "test_split_evaluation": results
        }, f, indent=4)
    print(f"\n✅ Saved real AWLF validation report to {report_path}")

if __name__ == "__main__":
    main()
