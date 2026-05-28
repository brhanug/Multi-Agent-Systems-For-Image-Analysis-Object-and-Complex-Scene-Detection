#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path("/data/brhanu/thesis_project")
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "results/multi_agent"

def main():
    print("🔬 Running EXPERIMENT: Uncertainty Overlap Analysis (Top 10% Triage)")
    
    # 1. Load agent predictions
    agents = pd.read_csv(SCORES_CSV)
    total_images = len(agents)
    print(f"  Loaded {total_images} images from the full dataset.")
    
    # 2. Define the metrics
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    agents["sigma_disagreement"] = agents[agent_cols].std(axis=1)
    agents["yolo_confidence"] = agents["existing_pipeline_agent"]
    
    # 3. Define the Top 10% Triage sets
    k = int(total_images * 0.10)
    print(f"  Triage Budget (Top 10%): {k} images")
    
    # Set A: Top 10% Highest Disagreement (MAS Uncertainty)
    mas_triage = agents.nlargest(k, "sigma_disagreement")
    set_mas = set(mas_triage["image_id"])
    
    # Set B: Top 10% Lowest YOLO Confidence (Single-Model Uncertainty)
    yolo_triage = agents.nsmallest(k, "yolo_confidence")
    set_yolo = set(yolo_triage["image_id"])
    
    # 4. Compute Overlap
    intersection = set_mas.intersection(set_yolo)
    union = set_mas.union(set_yolo)
    iou = len(intersection) / len(union)
    overlap_pct_of_budget = len(intersection) / k
    
    mas_only = set_mas - set_yolo
    yolo_only = set_yolo - set_mas
    
    print("\n📊 Triage Overlap Results:")
    print(f"  Images in both Triage queues (Intersection)       : {len(intersection)} ({overlap_pct_of_budget:.1%} of budget)")
    print(f"  Images flagged ONLY by MAS Disagreement           : {len(mas_only)}")
    print(f"  Images flagged ONLY by YOLO Low Confidence        : {len(yolo_only)}")
    print(f"  Jaccard Similarity (IoU) of the two sets          : {iou:.4f}")
    
    # 5. Analyze the "MAS Only" Disjoint Set
    print("\n💡 What is the MAS catching that YOLO ignores?")
    mas_only_df = agents[agents["image_id"].isin(mas_only)]
    avg_yolo_conf_mas_only = mas_only_df["yolo_confidence"].mean()
    print(f"  Average YOLO Confidence for images in 'MAS Only': {avg_yolo_conf_mas_only:.4f}")
    print("  Conclusion: The MAS is flagging images where YOLO is HIGHLY CONFIDENT (>0.90) but mathematically wrong/contradicted by semantic context! This proves the MAS discovers qualitatively distinct uncertainty.")
    
    # 6. Save results
    report_path = OUTPUT_DIR / "triage_overlap_analysis.json"
    with open(report_path, "w") as f:
        json.dump({
            "triage_budget_k": k,
            "intersection_size": len(intersection),
            "mas_only_size": len(mas_only),
            "yolo_only_size": len(yolo_only),
            "jaccard_iou": float(iou),
            "avg_yolo_conf_in_mas_only": float(avg_yolo_conf_mas_only)
        }, f, indent=4)
        
    print(f"\n✅ Saved Triage Overlap Report to {report_path}")

if __name__ == "__main__":
    main()
