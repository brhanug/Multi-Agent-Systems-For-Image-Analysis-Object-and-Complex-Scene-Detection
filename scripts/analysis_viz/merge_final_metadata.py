#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge auxiliary metadata into manifest.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root directory",
    )
    parser.add_argument(
        "--dataset-root",
        default="final_dataset",
        help="Dataset root directory (relative to base-dir or absolute path)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dataset_root_arg = Path(args.dataset_root)
    DATASET_ROOT = dataset_root_arg if dataset_root_arg.is_absolute() else base_dir / dataset_root_arg
    MANIFEST_PATH = DATASET_ROOT / "metadata" / "manifest.csv"
    KOSMOS_PATH = base_dir / "results" / "kosmos_grounding.jsonl"
    STUDENT_PATH = base_dir / "results" / "aligned_detections" / "student_iter_3_aligned.json"
    VQA_PATH = DATASET_ROOT / "metadata" / "vqa_binary_classification.json"
    SRS_PATH = DATASET_ROOT / "metadata" / "srs_scores.json"

    print("📥 Loading manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Load Kosmos
    kosmos_dict = {}
    if KOSMOS_PATH.exists():
        print("🔍 Loading Kosmos-2.5 data...")
        with open(KOSMOS_PATH, "r") as f:
            for line in f:
                obj = json.loads(line)
                # Key is "image", value is "kosmos_output"
                # Strip extension for matching image_id if image_id doesn't have it
                img_id = os.path.splitext(obj["image"])[0]
                kosmos_dict[img_id] = obj.get("kosmos_output", "")
    
    # Load Student 3 Detections
    student_dict = {}
    if STUDENT_PATH.exists():
        print("🔍 Loading Student Iteration 3 detections...")
        with open(STUDENT_PATH, "r") as f:
            student_data = json.load(f)
            for img_id, dets in student_data.items():
                student_dict[img_id] = len(dets)
                
    # New Columns
    df["kosmos_markdown"] = df["image_id"].apply(lambda x: kosmos_dict.get(x.split("/")[-1], ""))
    df["student_v3_count"] = df["image_id"].apply(lambda x: student_dict.get(x.split("/")[-1], 0))

    # SAVE
    output_path = DATASET_ROOT / "metadata" / "manifest_v2.csv"
    print(f"💾 Saving final manifest to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Overwrite original if successful
    df.to_csv(MANIFEST_PATH, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
