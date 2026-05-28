#!/usr/bin/env python3
import sqlite3
import json
import os
import shutil
import pandas as pd
from pathlib import Path

# Paths
BASE = Path("/data/brhanu/thesis_project")
DB_PATH = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")
IMG_SRC_DIR = BASE / "cvat_upload_temp/obj_train_data"
AUDIT_IMG_DIR = BASE / "human_spatial_audit/images"
AUDIT_LBL_DIR = BASE / "human_spatial_audit/labels"
OUTPUT_CSV = BASE / "human_spatial_audit/user_annotations_200.csv"

# YOLO 10-class mapping
CLASS_MAPPING = {
    'person': 0,
    'child': 1,
    'horse': 2,
    'building': 3,
    'weapon': 4,
    'vehicle': 5,
    'tree': 6,
    'clothing': 7,
    'text': 8,
    'animal': 9
}

def clean_and_make_dirs():
    print("🧹 Cleaning and recreating spatial audit directories...")
    if AUDIT_IMG_DIR.exists():
        shutil.rmtree(AUDIT_IMG_DIR)
    if AUDIT_LBL_DIR.exists():
        shutil.rmtree(AUDIT_LBL_DIR)
    os.makedirs(AUDIT_IMG_DIR, exist_ok=True)
    os.makedirs(AUDIT_LBL_DIR, exist_ok=True)

def main():
    clean_and_make_dirs()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query tasks and completions for task_id <= 200
    cursor.execute("""
        SELECT t.id, t.data, tc.result 
        FROM task t
        JOIN task_completion tc ON t.id = tc.task_id
        WHERE t.id <= 200
        ORDER BY t.id ASC
    """)
    
    rows = cursor.fetchall()
    print(f"Loaded {len(rows)} tasks from SQLite.")
    
    parsed_records = []
    
    for task_id, data_str, result_str in rows:
        data = json.loads(data_str)
        result = json.loads(result_str)
        
        # Get filename from URL
        img_url = data.get('image', '')
        filename = img_url.split('/')[-1]
        
        src_img_path = IMG_SRC_DIR / filename
        dest_img_path = AUDIT_IMG_DIR / filename
        
        if not src_img_path.exists():
            print(f"⚠️ Image not found in source directory: {filename}")
            continue
            
        # Copy image
        shutil.copy(src_img_path, dest_img_path)
        
        # Parse annotations
        bboxes = []
        scenes = []
        confidence = "5 - Certain"  # default
        clarity = "Clear"          # default
        
        for item in result:
            from_name = item.get('from_name')
            item_type = item.get('type')
            value = item.get('value', {})
            
            if item_type == 'rectanglelabels':
                labels = value.get('rectanglelabels', [])
                if not labels:
                    continue
                label = labels[0].lower()
                
                # Check class mapping
                if label in CLASS_MAPPING:
                    class_idx = CLASS_MAPPING[label]
                    
                    # Convert to YOLO coordinates
                    # LS: x, y, width, height are in percentages (0-100)
                    w_norm = value.get('width', 0) / 100.0
                    h_norm = value.get('height', 0) / 100.0
                    x_center = (value.get('x', 0) + value.get('width', 0) / 2.0) / 100.0
                    y_center = (value.get('y', 0) + value.get('height', 0) / 2.0) / 100.0
                    
                    bboxes.append((class_idx, x_center, y_center, w_norm, h_norm))
                else:
                    # Ignore other classes
                    pass
            elif item_type == 'choices':
                choices = value.get('choices', [])
                if choices:
                    choice_val = choices[0]
                    if from_name == 'scene':
                        scenes.append(choice_val.lower())
                    elif from_name == 'confidence':
                        confidence = choice_val
                    elif from_name == 'clarity':
                        clarity = choice_val
        
        # Write YOLO label file
        label_filename = filename.rsplit('.', 1)[0] + '.txt'
        label_path = AUDIT_LBL_DIR / label_filename
        
        with open(label_path, 'w') as f:
            for box in bboxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
        # Record details
        parsed_records.append({
            'task_id': task_id,
            'image_id': filename.rsplit('.', 1)[0],
            'filename': filename,
            'scene': scenes[0] if scenes else '',
            'confidence': confidence,
            'clarity': clarity,
            'num_boxes': len(bboxes)
        })
        
    df = pd.DataFrame(parsed_records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"🎉 Successfully processed {len(df)} images.")
    print(f"📷 Copied images to: {AUDIT_IMG_DIR}")
    print(f"🏷️ Created labels in: {AUDIT_LBL_DIR}")
    print(f"📊 Saved parsed CSV metadata to: {OUTPUT_CSV}")
    
    conn.close()

if __name__ == '__main__':
    main()
