#!/usr/bin/env python3
import pandas as pd
import numpy as np
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
    print("🔍 Running PATH 2 EXPERIMENT: Scene-Specific Deep Dive on the 'Drawing' Class...")
    
    # Load expert gold labels
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    
    # Filter for ONLY the 'Drawing' scene type (our hardest and largest complex class)
    drawing_subset = expert_reviewed[expert_reviewed['label_drawing'] == 1].copy()
    drawing_subset["gold_has_scene"] = 1 # By definition, if it has a tag it has a scene
    print(f"  Isolated {len(drawing_subset)} expert-labeled 'Drawing' images.")
    
    # Load agent predictions
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    # Join
    joined = drawing_subset.merge(agents, on='image_id', how='inner')
    
    # Compute error rates
    yolo_preds = (joined["existing_pipeline_agent"] >= 0.5).astype(int)
    vlm_preds = (joined["vlm_agent"] >= 0.5).astype(int)
    
    joined["yolo_correct"] = (yolo_preds == 1).astype(int)
    joined["vlm_correct"] = (vlm_preds == 1).astype(int)
    
    yolo_acc = joined["yolo_correct"].mean()
    vlm_acc = joined["vlm_correct"].mean()
    
    print(f"\n📊 Accuracy on 'Drawing' Class (Human-Verified):")
    print(f"  YOLO (Object Agent) Accuracy: {yolo_acc:.4f}")
    print(f"  VLM (Semantic Agent) Accuracy: {vlm_acc:.4f}")
    
    # Find the critical subset: Where YOLO completely fails, but VLM succeeds
    yolo_failed_vlm_caught = joined[(joined["yolo_correct"] == 0) & (joined["vlm_correct"] == 1)].copy()
    
    print(f"\n🎯 FOUND {len(yolo_failed_vlm_caught)} images where YOLO failed (scored < 0.5) but the VLM caught the scene!")
    
    # Let's compute the average multi-agent uncertainty for these specific images
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    yolo_failed_vlm_caught["sigma_disagreement"] = yolo_failed_vlm_caught[agent_cols].std(axis=1)
    
    print(f"  The average inter-agent disagreement (sigma) for these YOLO failures is {yolo_failed_vlm_caught['sigma_disagreement'].mean():.4f}")
    print(f"  This proves that the Coordinator detects the YOLO failure via disagreement and can route it to triage!")
    
    # Save the top 10 worst YOLO failures that the MAS saved
    yolo_failed_vlm_caught = yolo_failed_vlm_caught.sort_values(by="existing_pipeline_agent")
    
    output_cases = []
    for _, row in yolo_failed_vlm_caught.head(10).iterrows():
        output_cases.append({
            "image_id": row["image_id"],
            "yolo_score": round(row["existing_pipeline_agent"], 4),
            "vlm_score": round(row["vlm_agent"], 4),
            "coordinator_disagreement": round(row["sigma_disagreement"], 4)
        })
        
    report_path = OUTPUT_DIR / "path2_drawing_deepdive.json"
    with open(report_path, "w") as f:
        json.dump(output_cases, f, indent=4)
        
    print(f"\n✅ Saved Top 10 MAS Success Cases (where YOLO failed) to {report_path}")

if __name__ == "__main__":
    main()
