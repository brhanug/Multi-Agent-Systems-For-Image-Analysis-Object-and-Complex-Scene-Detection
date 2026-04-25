#!/usr/bin/env python3
import os
import csv
from pathlib import Path

# === CONFIG ===
PACKAGE_DIR = "/data/brhanu/thesis_project/final_dataset_v1_refresh"
MANIFEST = os.path.join(PACKAGE_DIR, "metadata/manifest_v1.csv")
IMAGES_DIR = os.path.join(PACKAGE_DIR, "images/restored")
LABELS_DIR = os.path.join(PACKAGE_DIR, "labels/yolo_labels")

def main():
    print("📋 Final v1.0 Integrity Check...")
    
    with open(MANIFEST, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"🔹 Checking {len(rows)} manifest entries...")
    
    missing_images = 0
    missing_labels = 0
    has_labels_count = 0
    
    for row in rows:
        img_id = row['image_id'] # e.g. PPN.../0000_1.jpg
        img_path = os.path.join(PACKAGE_DIR, "images", "restored", img_id)
        
        if not os.path.exists(img_path):
            missing_images += 1
            
        if row['has_label'] == "Yes":
            has_labels_count += 1
            # Check label file
            label_fname = img_id.replace("/", "_").replace(".jpg", ".txt")
            if not os.path.exists(os.path.join(LABELS_DIR, label_fname)):
                missing_labels += 1

    print("\n--- RESULTS ---")
    print(f"Total Entries: {len(rows)}")
    print(f"Images Missing: {missing_images}")
    print(f"Labels Expected: {has_labels_count}")
    print(f"Labels Missing: {missing_labels}")
    
    if missing_images == 0 and missing_labels == 0:
        print("\n✅ PASSED: Package is structurally sound and consistent.")
    else:
        print("\n❌ FAILED: Found inconsistencies. See details above.")

if __name__ == "__main__":
    main()
