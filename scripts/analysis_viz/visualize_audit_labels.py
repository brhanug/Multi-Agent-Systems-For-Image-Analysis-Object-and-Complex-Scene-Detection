#!/usr/bin/env python3
import cv2
import os
import pandas as pd
from tqdm import tqdm

# Configuration
AUDIT_DIR = "/data/brhanu/thesis_project/human_baseline_gold_v2"
IMAGE_DIR = os.path.join(AUDIT_DIR, "images")
LABEL_DIR = os.path.join(AUDIT_DIR, "labels")
VIS_DIR = os.path.join(AUDIT_DIR, "visual_verification")

TAXONOMY = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text", "hat", "furniture"]
COLORS = [
    (255, 0, 0),   # person - Blue
    (0, 255, 0),   # child - Green
    (0, 0, 255),   # animal - Red
    (255, 255, 0), # building - Cyan
    (255, 0, 255), # weapon - Magenta
    (0, 255, 255), # vehicle - Yellow
    (128, 0, 0),   # tree - Maroon
    (0, 128, 0),   # text - Dark Green
    (0, 0, 128),   # hat - Navy
    (128, 128, 0)  # furniture - Olive
]

def visualize():
    os.makedirs(VIS_DIR, exist_ok=True)
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    
    print(f"🎨 Visualizing {len(images)} audit images...")
    for img_file in tqdm(images):
        img_path = os.path.join(IMAGE_DIR, img_file)
        lbl_file = img_file.replace(".jpg", ".txt")
        lbl_path = os.path.join(LABEL_DIR, lbl_file)
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convert YOLO to pixel
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = COLORS[cls_id % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = TAXONOMY[cls_id]
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(os.path.join(VIS_DIR, img_file), img)

if __name__ == "__main__":
    visualize()
