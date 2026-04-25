#!/usr/bin/env python3
import pandas as pd
import os
import shutil

# Configuration
BASELINE_CSV = "/data/brhanu/thesis_project/human_baseline_gold_v2/baseline_audit_manifest.csv"
DATASET_ROOT = "/data/brhanu/thesis_project/final_dataset"
OUTPUT_DIR = "/data/brhanu/thesis_project/human_baseline_gold_v2"
YOLO_LABELS_SRC = os.path.join(DATASET_ROOT, "labels/yolo_labels")

def prepare_audit_kit():
    print(f"🚀 Loading baseline manifest: {BASELINE_CSV}")
    df = pd.read_csv(BASELINE_CSV)
    
    img_out = os.path.join(OUTPUT_DIR, "images")
    lbl_out = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    
    count = 0
    missing_img = 0
    missing_lbl = 0
    
    for _, row in df.iterrows():
        img_id = row['image_id']
        safe_id = img_id.replace('/', '_')
        
        # 1. Copy Image
        rel_path = row['restored_path']
        src_img = os.path.join(DATASET_ROOT, rel_path)
        dst_img = os.path.join(img_out, safe_id + ".jpg")
        
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            count += 1
        else:
            missing_img += 1
            
        # 2. Copy YOLO Label (if exists)
        src_lbl = os.path.join(YOLO_LABELS_SRC, safe_id + ".txt")
        dst_lbl = os.path.join(lbl_out, safe_id + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            missing_lbl += 1
        
        if count % 100 == 0:
            print(f"  Processed {count} images...")
            
    print(f"✅ Audit Kit preparation complete.")
    print(f"  Success (Images): {count}")
    print(f"  Missing Images: {missing_img}")
    print(f"  Missing Labels: {missing_lbl}")
    print(f"📦 Scene labels available in: {BASELINE_CSV} (vqa_primary_scene column)")

if __name__ == "__main__":
    prepare_audit_kit()
