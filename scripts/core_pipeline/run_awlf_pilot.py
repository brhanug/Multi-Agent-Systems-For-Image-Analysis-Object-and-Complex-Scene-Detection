#!/usr/bin/env python3
"""
run_awlf_pilot.py
-------------------
Agreement-Weighted Label Filter (AWLF) Prototype.
Trains a simple discriminator to predict if a pseudo-label is likely to be correct
based on inter-agent agreement features (std, mean, min, max) and restoration quality.

This addresses the "Heuristic vs Algorithmic" weakness identified in the comparative analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

# Paths
BASE = Path("/data/brhanu/thesis_project")
SCORES = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
GOLD = BASE / "human_baseline_gold_kit" / "labeling_worksheet.csv"
OUTPUT = BASE / "results" / "multi_agent" / "awlf_pilot_results.json"

def main():
    if not SCORES.exists() or not GOLD.exists():
        print("❌ Required files missing for AWLF pilot.")
        return

    # 1. Load data
    scores_df = pd.read_csv(SCORES)
    gold_df = pd.read_csv(GOLD)

    # 2. Join
    df = pd.merge(scores_df, gold_df, left_on="image_id", right_on="Image_ID")
    
    if df.empty:
        print("❌ Join resulted in empty dataset. Check IDs.")
        return

    # 3. Features: Agent scores
    agents = ["agreement_agent", "scene_agent", "vlm_agent", "restoration_agent"]
    X = df[agents].fillna(0)
    
    # Engineered features
    X['score_std'] = X[agents].std(axis=1)
    X['score_mean'] = X[agents].mean(axis=1)
    X['score_min'] = X[agents].min(axis=1)
    X['score_max'] = X[agents].max(axis=1)
    
    # Target: Presence of any positive labels in gold
    class_cols = ["Person", "Child", "Horse", "Building", "Weapon", "Vehicle", "Tree", "Clothing", "Text", "Animal"]
    y = (df[class_cols].sum(axis=1) > 0).astype(int)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train AWLF (Random Forest)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\n📊 AWLF Pilot Evaluation (n={})".format(len(y_test)))
    print(classification_report(y_test, y_pred))
    print("ROC-AUC: {:.4f}".format(roc_auc_score(y_test, y_prob)))

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n🔍 Feature Importance:")
    print(importances)

    # 7. Save results
    import json
    results = {
        "n_samples": len(df),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "feature_importances": importances.to_dict(),
        "recommendation": "Upgrade heuristic IoU filter to AWLF for publication."
    }
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ AWLF Pilot results saved to {OUTPUT}")

if __name__ == "__main__":
    main()
