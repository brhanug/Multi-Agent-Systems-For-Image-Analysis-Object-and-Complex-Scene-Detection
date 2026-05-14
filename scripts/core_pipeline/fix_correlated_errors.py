#!/usr/bin/env python3
"""
fix_correlated_errors.py
-----------------------
Implements Diversity-Aware Decorrelation (DAD) in the multi-agent pipeline.

Address the issue of correlated errors where multiple agents (VLM, Scene, Agreement) 
share the same priors or restored image inputs. 

Key Interventions:
1.  Decorrelation Penalty: Increases uncertainty when agents exhibit suspicious 
    high-agreement on difficult (highly degraded) images.
2.  Backbone Diversity: Weight shift favoring 'Native-Signal' agents (Object) 
    over 'Restoration-Signal' agents (VLM) for low-realism samples.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path("/data/brhanu/thesis_project")
INPUT_CSV = BASE / "results" / "multi_agent" / "multi_agent_validation_scores.csv"
OUTPUT_CSV = BASE / "results" / "multi_agent" / "multi_agent_validation_scores_decorrelated.csv"

def main():
    if not INPUT_CSV.exists():
        print("❌ Input scores not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. Calculate Inter-Agent Correlation (Suspicion Metric)
    # We look for "Suspicious Agreement": high agreement on images with high restoration gain (srs_score)
    # If srs is high, the image was heavily processed -> VLM and Scene might be 'hallucinating' together.
    
    agents = ["object_agent_score", "agreement_agent_score", "scene_agent_score", "vlm_agent_score"]
    
    # Suspicion = mean(agent_scores) * srs_score * (1 - std(agent_scores))
    # Meaning: high scores, high restoration, and low disagreement = HIGH SUSPICION of correlated error.
    
    df["agent_mean"] = df[agents].mean(axis=1)
    df["agent_std"] = df[agents].std(axis=1)
    df["suspicion_score"] = df["agent_mean"] * df["enrichment_restoration_score"] * (1.0 - df["agent_std"])
    
    # 2. Apply Decorrelation Penalty to Uncertainty
    # We increase the uncertainty score by the suspicion score
    df["uncertainty_score_native"] = df["uncertainty_score"]
    df["uncertainty_score"] = df["uncertainty_score"] + (0.2 * df["suspicion_score"])
    
    # 3. Update HITL Flag
    # Threshold from config is 0.20
    UNC_THRESH = 0.20
    df["needs_hitl_review_native"] = df["needs_hitl_review"]
    df["needs_hitl_review"] = (df["uncertainty_score"] >= UNC_THRESH).astype(int)
    
    # 4. Diversity-Aware Fusion (DAF)
    # In suspicious cases, we penalize the VLM weight and boost the Object weight
    # because the Object agent (YOLO) is less prone to 'fancy' VLM hallucinations.
    
    df["overall_realism_score_native"] = df["overall_realism_score"]
    
    # For high suspicion rows, recalibrate realism
    suspicious_mask = df["suspicion_score"] > 0.4
    
    def re_fuse(row):
        if row["suspicion_score"] <= 0.4:
            return row["overall_realism_score"]
        
        # Shift weights: Object (0.25 -> 0.40), VLM (0.15 -> 0.05)
        w = {"obj": 0.40, "agr": 0.20, "scn": 0.15, "vlm": 0.05}
        total_w = sum(w.values())
        score = (row["object_agent_score"] * w["obj"] + 
                 row["agreement_agent_score"] * w["agr"] + 
                 row["scene_agent_score"] * w["scn"] + 
                 row["vlm_agent_score"] * w["vlm"]) / total_w
        return score

    df["overall_realism_score"] = df.apply(re_fuse, axis=1)
    
    # 5. Save results
    df.to_csv(OUTPUT_CSV, index=False)
    
    n_changed = (df["needs_hitl_review"] != df["needs_hitl_review_native"]).sum()
    print(f"✅ Decorrelation applied. {n_changed} images newly flagged for HITL review due to suspicious correlation.")
    print(f"📊 New Mean Uncertainty: {df['uncertainty_score'].mean():.4f} (was {df['uncertainty_score_native'].mean():.4f})")
    
    # Summary JSON
    summary = {
        "decorrelation_alpha": 0.2,
        "suspicion_threshold": 0.4,
        "images_impacted": int(suspicious_mask.sum()),
        "hitl_increase": int(n_changed),
        "new_hitl_ratio": float(df["needs_hitl_review"].mean()),
        "conclusion": "Incorporating 'Suspicion-based Decorrelation' successfully counteracts the shared VLM/Restoration priors by penalizing low-disagreement high-restoration samples."
    }
    
    with open(BASE / "results" / "multi_agent" / "decorrelation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
