#!/usr/bin/env python3
"""
prepare_annotator_worksheets.py
Generates the new Gold-Label dataset worksheets with 1-5 scale ambiguity metrics.
Creates simulated data for two annotators to test Cohen's Kappa logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

BASE = Path(__file__).resolve().parents[2]
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "human_baseline_gold_kit"

def simulate_human_scores(sigma, base_noise=0.5):
    """
    Simulates a human 1-5 score loosely correlated with the agent disagreement (sigma).
    High sigma -> High ambiguity (1-5).
    """
    # Normalize sigma roughly to 0-1 (assuming max is around 0.4)
    norm_sigma = np.clip(sigma / 0.4, 0, 1)
    
    # Base expected score (1 to 5)
    expected_ambiguity = 1 + (norm_sigma * 4)
    
    # Add annotator noise
    noise = np.random.normal(0, base_noise)
    final_score = np.clip(np.round(expected_ambiguity + noise), 1, 5)
    return int(final_score)

def main():
    if not SCORES_CSV.exists():
        print(f"❌ Error: {SCORES_CSV} not found.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading agent comparison scores to sample 300 images...")
    df = pd.read_csv(SCORES_CSV)
    
    # Calculate sigma to drive the simulation
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    df["sigma"] = df[agent_cols].std(axis=1).fillna(0)
    
    # Sample 300
    df_sample = df.sample(n=300, random_state=42).copy()
    
    # Columns we want in the worksheet
    columns = [
        "Image_ID", "Person", "Child", "Building", "Horse", 
        "Ambiguity_Score_1_to_5", "Interpretability_1_to_5", "Authenticity_1_to_5",
        "Primary_Scene", "Notes"
    ]
    
    annotator_A = []
    annotator_B = []
    
    np.random.seed(101)
    
    for _, row in df_sample.iterrows():
        img_id = row["image_id"]
        sigma = row["sigma"]
        
        # Simulate binary objects (Person, Child, etc) roughly based on some random chance
        # We want A and B to agree most of the time
        obj_probs = np.random.rand(4)
        obj_A = (obj_probs > 0.5).astype(int)
        
        # Annotator B flips 10% of the time
        flips = (np.random.rand(4) > 0.9).astype(int)
        obj_B = np.abs(obj_A - flips)
        
        # Simulate 1-5 scales
        amb_A = simulate_human_scores(sigma, base_noise=0.5)
        amb_B = simulate_human_scores(sigma, base_noise=0.7) # B is slightly noisier
        
        # Interpretability is roughly inverse to Ambiguity
        int_A = 6 - simulate_human_scores(sigma, base_noise=0.6)
        int_B = 6 - simulate_human_scores(sigma, base_noise=0.6)
        
        auth_A = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        auth_B = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        
        annotator_A.append([
            img_id, obj_A[0], obj_A[1], obj_A[2], obj_A[3],
            amb_A, int_A, auth_A, "landscape", "[SIMULATED_A]"
        ])
        
        annotator_B.append([
            img_id, obj_B[0], obj_B[1], obj_B[2], obj_B[3],
            amb_B, int_B, auth_B, "landscape", "[SIMULATED_B]"
        ])

    df_A = pd.DataFrame(annotator_A, columns=columns)
    df_B = pd.DataFrame(annotator_B, columns=columns)
    
    out_A = OUTPUT_DIR / "annotator_A_worksheet.csv"
    out_B = OUTPUT_DIR / "annotator_B_worksheet.csv"
    
    df_A.to_csv(out_A, index=False)
    df_B.to_csv(out_B, index=False)
    
    print(f"✅ Generated {len(df_A)} rows for {out_A.name}")
    print(f"✅ Generated {len(df_B)} rows for {out_B.name}")

if __name__ == "__main__":
    main()
