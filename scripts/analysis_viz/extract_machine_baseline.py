import os
import pandas as pd
from ultralytics import YOLO
import json

# Paths
SUBSET_DIR = "/data/brhanu/thesis_project/human_baseline_subset"
MODEL_PATH = "/data/brhanu/thesis_project/results/yolo11_final_v1_refresh/exp_final2/weights/best.pt"
OUTPUT_PATH = "/data/brhanu/thesis_project/results/human_baseline_machine_results.json"

# Classes
CLASS_MAP = {0: 'person', 1: 'child', 2: 'animal', 3: 'building', 4: 'weapon', 5: 'vehicle', 6: 'tree', 7: 'text', 8: 'hat', 9: 'furniture'}

def run_machine_baseline():
    print(f"🚀 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    images = [f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')]
    results_data = {}
    
    print(f"🔍 Processing {len(images)} images for machine baseline...")
    for img_name in images:
        img_path = os.path.join(SUBSET_DIR, img_name)
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        preds = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                preds.append({"class": CLASS_MAP[cls_id], "conf": round(conf, 3)})
        
        results_data[img_name] = preds

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✅ Machine baseline results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_machine_baseline()
