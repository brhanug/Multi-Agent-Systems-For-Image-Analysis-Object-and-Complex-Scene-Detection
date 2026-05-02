#!/usr/bin/env python3
"""
Active Agent Adjudicator (Phase 5).

This script identifies high-disagreement images where base agents conflict
(e.g., YOLO detects a horse, but GroundingDINO detects a child) and formulates
a prompt for the LLaVA-OneVision VLM to act as the adjudicator.

Instead of a blind weighted average, the VLM actively resolves the dispute.

Usage:
  This generates the prompts. You can then pipe the output to your LLaVA
  inference script (e.g., `run_llava_vqa.py`) to get the final answers.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def find_disputes(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Find images where the object agent and scene agent (or VLM) heavily disagree.
    In a real system, you'd compare the actual string labels (e.g., "horse" vs "dog").
    Here we flag images with high variance in agent scores.
    """
    # For demonstration, we simulate dispute detection by finding cases where 
    # one agent scores high (>0.7) and another scores low (<0.3).
    disputes = []
    
    for _, row in df.iterrows():
        img = row.get("image_id")
        obj = row.get("existing_pipeline_agent", 0.0)
        scn = row.get("scene_agent", 0.0)
        
        # Simulated dispute condition: object detector found something confident, 
        # but scene classifier disagrees entirely with the context.
        if abs(obj - scn) > threshold:
            disputes.append({
                "image_id": img,
                "object_score": obj,
                "scene_score": scn,
                "dispute_type": "Object vs Scene",
                "prompt": (
                    f"Agent A (Object Detector) is highly confident about its detection (score: {obj:.2f}), "
                    f"but Agent B (Scene Classifier) disagrees with the context (score: {scn:.2f}). "
                    "Look carefully at this historical image. Is the primary object correctly identified in a "
                    "plausible historical context? Who is right and why?"
                )
            })
            
    return pd.DataFrame(disputes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output", default="results/multi_agent/adjudication_prompts.json")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    csv_path = base / args.scores_csv if not Path(args.scores_csv).is_absolute() else Path(args.scores_csv)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    disputes_df = find_disputes(df, threshold=0.5)
    
    out_path = base / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = disputes_df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    print(f"✅ Found {len(records)} high-disagreement cases.")
    print(f"✅ Wrote adjudication prompts to {out_path}")
    if records:
        print("\nExample prompt:")
        print(f"Image: {records[0]['image_id']}")
        print(f"Prompt: {records[0]['prompt']}")

if __name__ == "__main__":
    main()
