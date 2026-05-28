#!/usr/bin/env python3
import pandas as pd
import shutil
from pathlib import Path
import random

BASE = Path("/data/brhanu/thesis_project")
GOLD_LABELS_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SOURCE_IMAGE_DIR = BASE / "data" / "images"  # Assuming original images are here, or in human_baseline_gold_kit/images
AUDIT_DIR = BASE / "human_spatial_audit"
AUDIT_IMAGES_DIR = AUDIT_DIR / "images"
AUDIT_LABELS_DIR = AUDIT_DIR / "labels"

def main():
    print("📦 Preparing 50-image Spatial Audit Kit...")
    
    # Create directories if they don't exist
    AUDIT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the 801 expert-labeled images
    df = pd.read_csv(GOLD_LABELS_CSV)
    df['cvat_id'] = df.index
    expert_reviewed = df[df['cvat_id'] <= 800].copy()
    
    # 2. Sample 50 random images
    sampled = expert_reviewed.sample(n=50, random_state=42).copy()
    print(f"  Selected 50 random images from the 801 expert subset.")
    
    # 3. Copy images to the audit folder
    # In the CSV, raw_name is typically the relative path or filename
    copied_count = 0
    missing_count = 0
    
    # Try different potential source directories
    potential_sources = [
        BASE / "cvat_upload_temp" / "obj_train_data",
        BASE / "human_baseline_gold_kit" / "images",
        BASE / "final_dataset" / "images",
        BASE / "data" / "images"
    ]
    
    for idx, row in sampled.iterrows():
        img_name = Path(row['raw_name']).name
        
        # Find the image
        src_path = None
        for src_dir in potential_sources:
            candidate = src_dir / img_name
            if candidate.exists():
                src_path = candidate
                break
                
        if src_path:
            dst_path = AUDIT_IMAGES_DIR / img_name
            shutil.copy2(src_path, dst_path)
            
            # Create an empty labels file for YOLO format readiness
            label_file = AUDIT_LABELS_DIR / (img_name.rsplit('.', 1)[0] + ".txt")
            label_file.touch()
            copied_count += 1
        else:
            missing_count += 1
            print(f"  ⚠️ Could not find image: {img_name}")
            
    print(f"\n✅ Successfully prepared {copied_count} images for spatial annotation in {AUDIT_IMAGES_DIR}")
    if missing_count > 0:
        print(f"  ⚠️ {missing_count} images were not found in the source directories.")
        
    print(f"\n📝 Next Steps:")
    print(f"  1. Open CVAT, LabelImg, or Label Studio.")
    print(f"  2. Load the images from {AUDIT_IMAGES_DIR}.")
    print(f"  3. Draw bounding boxes for the core classes.")
    print(f"  4. Export the labels in YOLO format to {AUDIT_LABELS_DIR}.")

if __name__ == "__main__":
    main()
