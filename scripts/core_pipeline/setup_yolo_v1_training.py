#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

# === CONFIG ===
BASE_DIR = "/data/brhanu/thesis_project"
LABEL_DIR = os.path.join(BASE_DIR, "final_dataset_v1_refresh/labels/yolo_labels")
IMAGE_ROOT = os.path.join(BASE_DIR, "final_dataset/images/restored")
YOLO_ROOT = os.path.join(BASE_DIR, "data/yolo11_v1_refresh")

TRAIN_SPLIT = 0.9

def main():
    print(f"🚀 Preparing YOLO training set at {YOLO_ROOT}...")
    
    # Setup folders
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(YOLO_ROOT, d), exist_ok=True)
    
    # Get all label files
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]
    random.shuffle(label_files)
    
    split_idx = int(len(label_files) * TRAIN_SPLIT)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]
    
    def process_split(files, split_name):
        print(f"📦 Processing {split_name} ({len(files)} files)...")
        for f in files:
            img_id = os.path.splitext(f)[0]
            # Map img_id (e.g. PPN..._0000_1) back to path (PPN.../0000_1.jpg)
            # My current img_id in labels is PPNXXXXX_YYYYYY_Z.txt
            # I need to find the actual image path.
            
            # Simple heuristic mapping:
            parts = img_id.split("_")
            ppn = parts[0]
            rest = "_".join(parts[1:])
            img_path = os.path.join(IMAGE_ROOT, ppn, f"{rest}.jpg")
            
            if not os.path.exists(img_path):
                 # Fallback search if heuristic fails
                 print(f"⚠️ Warning: Could not find image for {img_id}, skipping.")
                 continue
            
            # Use absolute symlinks
            dst_img = os.path.join(YOLO_ROOT, "images", split_name, f"{img_id}.jpg")
            dst_lbl = os.path.join(YOLO_ROOT, "labels", split_name, f)
            
            if os.path.exists(dst_img): os.remove(dst_img)
            if os.path.exists(dst_lbl): os.remove(dst_lbl)
            
            os.symlink(img_path, dst_img)
            os.symlink(os.path.join(LABEL_DIR, f), dst_lbl)

    process_split(train_files, "train")
    process_split(val_files, "val")
    
    print("✅ YOLO training set ready.")

if __name__ == "__main__":
    main()
