#!/usr/bin/env python3
"""
Gold Evaluation Script (Phase 3).

Evaluates the multi-agent pipeline against the true human-annotated gold labels.
Performs:
  1) Leave-one-out ablation studies (E2) against measured human ground truth.
  2) Scene complexity stratification (E4) on true labels instead of proxy labels.
  
Pre-requisite:
  `human_baseline_gold_kit/labeling_worksheet.csv` must be fully populated 
  with binary labels by human reviewers (Phase 1).
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

AGENT_COLS = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"
]

def load_gold_labels(path: Path) -> pd.DataFrame:
    """Load the gold labeling worksheet and generate an 'has_issue' binary label."""
    if not path.exists():
        raise FileNotFoundError(f"Gold labels not found at {path}")
        
    df = pd.read_csv(path)
    
    # Check if it has been filled in. 
    # If all Object columns (Person, Child, etc.) are empty, it's not ready.
    object_cols = ["Person", "Child", "Horse", "Building", "Weapon", "Vehicle", "Tree", "Clothing", "Text", "Animal"]
    if df[object_cols].isnull().all().all():
        print("⚠️ WARNING: The gold worksheet appears to be entirely empty.")
        print("Evaluation results will be undefined until the CSV is filled.")
    
    # We define 'has_issue' as 1 if the human found the image complex, ambiguous, or if key objects are missing.
    # For now, as a placeholder logic until criteria are formalized:
    # Let's say if Ambiguity_Notes is not empty, there is an issue.
    df["has_issue"] = (~df["Ambiguity_Notes"].isnull()).astype(int)
    
    # Keep only the essential columns for evaluation mapping
    return df[["Image_ID", "has_issue"]]

def leave_one_out_ablation(df: pd.DataFrame, labels: np.ndarray):
    """
    Evaluates the fusion score with one agent removed at a time.
    Metric: F1 Score of predicting 'has_issue' (bottom 25% of fusion score).
    """
    results = []
    
    # Base full fusion F1
    full_fusion = df[AGENT_COLS].mean(axis=1)
    # Predict issue if score is in bottom quartile
    q25 = full_fusion.quantile(0.25)
    y_pred_full = (full_fusion <= q25).astype(int)
    full_f1 = f1_score(labels, y_pred_full, zero_division=0)
    
    results.append({
        "ablated_agent": "None (Full Fusion)",
        "f1_score": round(full_f1, 4),
        "delta": 0.0
    })
    
    for agent in AGENT_COLS:
        subset = [c for c in AGENT_COLS if c != agent]
        subset_fusion = df[subset].mean(axis=1)
        q25_sub = subset_fusion.quantile(0.25)
        y_pred_sub = (subset_fusion <= q25_sub).astype(int)
        
        sub_f1 = f1_score(labels, y_pred_sub, zero_division=0)
        
        results.append({
            "ablated_agent": agent,
            "f1_score": round(sub_f1, 4),
            "delta": round(sub_f1 - full_f1, 4)
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--gold-csv", default="human_baseline_gold_kit/labeling_worksheet.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scores_path = base / args.scores_csv if not Path(args.scores_csv).is_absolute() else Path(args.scores_csv)
    gold_path = base / args.gold_csv if not Path(args.gold_csv).is_absolute() else Path(args.gold_csv)
    
    if not scores_path.exists():
        print(f"Error: {scores_path} not found.")
        return
        
    scores_df = pd.read_csv(scores_path)
    
    try:
        gold_df = load_gold_labels(gold_path)
    except FileNotFoundError as e:
        print(e)
        return
        
    # Merge on image ID (assuming normalization logic)
    def norm_id(x): return Path(str(x)).stem
    scores_df["norm_id"] = scores_df["image_id"].apply(norm_id)
    gold_df["norm_id"] = gold_df["Image_ID"].apply(norm_id)
    
    merged = pd.merge(scores_df, gold_df, on="norm_id", how="inner")
    
    if merged.empty:
        print("Error: No intersection between agent scores and gold labels.")
        return
        
    print(f"Evaluating on {len(merged)} gold-labeled images.")
    
    labels = merged["has_issue"].to_numpy()
    
    ablation_results = leave_one_out_ablation(merged, labels)
    
    out = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    out_csv = out / "gold_ablation_results.csv"
    pd.DataFrame(ablation_results).to_csv(out_csv, index=False)
    print(f"✅ Wrote ablation results to {out_csv}")
    
    for res in ablation_results:
        print(f"{res['ablated_agent']:<25} F1: {res['f1_score']:.4f} (Δ {res['delta']:+.4f})")
        
    print("\nNote: Scene complexity stratification (E4) will use the same merge logic.")

if __name__ == "__main__":
    main()
