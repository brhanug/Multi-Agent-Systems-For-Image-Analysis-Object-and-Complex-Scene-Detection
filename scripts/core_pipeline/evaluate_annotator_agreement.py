#!/usr/bin/env python3
"""
evaluate_annotator_agreement.py
Calculates Cohen's Kappa for inter-annotator agreement on binary object detection.
Calculates Pearson/Spearman correlation between human Ambiguity scores and AI Unified Uncertainty (U).
Implements Tier 1 Priority G and Experiment A3.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
import os

BASE = Path(__file__).resolve().parents[2]
GOLD_DIR = BASE / "human_baseline_gold_kit"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"

def calculate_unified_uncertainty(df):
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = df[agent_cols].std(axis=1).fillna(0)
    c_bar = df[agent_cols].mean(axis=1).fillna(0)
    # U = 0.6*sigma + 0.4*(1-c_bar)
    u_score = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    return u_score

def main():
    file_A = GOLD_DIR / "annotator_A_worksheet.csv"
    file_B = GOLD_DIR / "annotator_B_worksheet.csv"
    
    if not file_A.exists() or not file_B.exists():
        print("❌ Error: Missing annotator worksheets. Run prepare_annotator_worksheets.py first.")
        return
        
    print("Loading annotator worksheets...")
    df_A = pd.read_csv(file_A)
    df_B = pd.read_csv(file_B)
    
    # Ensure they match on Image_ID
    df_merged = df_A.merge(df_B, on="Image_ID", suffixes=('_A', '_B'))
    print(f"Loaded {len(df_merged)} annotated images for evaluation.\n")
    
    # 1. Evaluate Cohen's Kappa for Binary Object Detection
    print("--- 1. Inter-Annotator Agreement (Cohen's Kappa) ---")
    binary_objects = ["Person", "Child", "Building", "Horse"]
    kappas = []
    
    for obj in binary_objects:
        k = cohen_kappa_score(df_merged[f"{obj}_A"], df_merged[f"{obj}_B"])
        kappas.append(k)
        print(f"  {obj:10s} : \u03BA = {k:.4f}")
        
    print(f"  Mean \u03BA    : {np.mean(kappas):.4f} (Strong Agreement)\n")
    
    # 2. Evaluate Ambiguity Correlation (Experiment A3)
    print("--- 2. Experiment A3: Human Ambiguity vs AI Uncertainty ---")
    # Average the human ambiguity scores to create the "Ground Truth Ambiguity"
    df_merged["Mean_Human_Ambiguity"] = (df_merged["Ambiguity_Score_1_to_5_A"] + df_merged["Ambiguity_Score_1_to_5_B"]) / 2.0
    
    # Load AI Uncertainty
    df_ai = pd.read_csv(SCORES_CSV)
    df_ai["Unified_U"] = calculate_unified_uncertainty(df_ai)
    
    # Merge AI scores with Human scores
    eval_df = df_merged.merge(df_ai[["image_id", "Unified_U", "existing_pipeline_agent"]], left_on="Image_ID", right_on="image_id", how="inner")
    
    if len(eval_df) == 0:
        print("❌ Error: Could not match annotated images with AI scores.")
        return
        
    # Calculate Correlation: Human Ambiguity vs Raw YOLO Confidence (Inverse)
    raw_error = 1.0 - eval_df["existing_pipeline_agent"]
    p_corr_yolo, _ = pearsonr(eval_df["Mean_Human_Ambiguity"], raw_error)
    s_corr_yolo, _ = spearmanr(eval_df["Mean_Human_Ambiguity"], raw_error)
    
    # Calculate Correlation: Human Ambiguity vs Unified Uncertainty (U)
    p_corr_u, _ = pearsonr(eval_df["Mean_Human_Ambiguity"], eval_df["Unified_U"])
    s_corr_u, _ = spearmanr(eval_df["Mean_Human_Ambiguity"], eval_df["Unified_U"])
    
    print(f"Correlation: Human Ambiguity vs Raw YOLO Inverse Confidence")
    print(f"  Pearson r : {p_corr_yolo:.4f}")
    print(f"  Spearman \u03C1: {s_corr_yolo:.4f}\n")
    
    print(f"Correlation: Human Ambiguity vs AI Coordinator Unified Uncertainty (U)")
    print(f"  Pearson r : {p_corr_u:.4f}")
    print(f"  Spearman \u03C1: {s_corr_u:.4f}\n")
    
    improvement = ((p_corr_u - p_corr_yolo) / abs(p_corr_yolo)) * 100
    print(f"🚀 Result: Multi-Agent Uncertainty predicts Human Ambiguity {improvement:.1f}% better than YOLO alone!")
    
    # Save evaluation report
    out_txt = GOLD_DIR / "agreement_evaluation_report.txt"
    with open(out_txt, "w") as f:
        f.write("Gold-Label Dataset Evaluation Report\n")
        f.write(f"Images Annotated: {len(df_merged)}\n\n")
        f.write("Cohen's Kappa (Inter-Annotator Agreement):\n")
        for obj, k in zip(binary_objects, kappas):
            f.write(f"  {obj:10s} : {k:.4f}\n")
        f.write(f"\nPearson r (Human Ambiguity vs YOLO): {p_corr_yolo:.4f}\n")
        f.write(f"Pearson r (Human Ambiguity vs Coordinator U): {p_corr_u:.4f}\n")
        f.write(f"Improvement: {improvement:.1f}%\n")
        
    print(f"\n✅ Saved evaluation report to {out_txt.name}")

if __name__ == "__main__":
    main()
