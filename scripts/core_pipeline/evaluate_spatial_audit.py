#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
from ultralytics import YOLO

BASE = Path("/data/brhanu/thesis_project")
AUDIT_DIR = BASE / "human_spatial_audit"
AUDIT_IMAGES = AUDIT_DIR / "images"
AUDIT_LABELS = AUDIT_DIR / "labels"
MODEL_WEIGHTS = BASE / "weights" / "yolo11_best.pt" # Update if exact weights name is different

def main():
    print("🎯 Initiating HUMAN SPATIAL AUDIT EVALUATION...")
    
    # 1. Check if labels exist and have content
    if not AUDIT_LABELS.exists():
        print(f"❌ Error: {AUDIT_LABELS} directory not found. Please complete the annotation first.")
        return
        
    labeled_files = [f for f in os.listdir(AUDIT_LABELS) if f.endswith('.txt') and os.path.getsize(AUDIT_LABELS / f) > 0]
    
    if len(labeled_files) == 0:
        print("⚠️ No bounding boxes found in the labels folder. Please finish drawing boxes in CVAT/LabelImg and save them.")
        return
        
    print(f"  Found {len(labeled_files)} images with manual human bounding box annotations!")
    
    # 2. Create a dynamic YAML dataset configuration for ultralytics
    yaml_path = AUDIT_DIR / "audit_dataset.yaml"
    
    dataset_yaml = {
        'path': str(AUDIT_DIR),
        'train': 'images', # Ultralytics requires a train split, we'll just point it here
        'val': 'images',
        'names': {
            0: 'person',
            1: 'child',
            2: 'horse',
            3: 'building',
            4: 'weapon',
            5: 'vehicle',
            6: 'tree',
            7: 'clothing',
            8: 'text',
            9: 'animal'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
        
    print(f"  Generated YOLO dataset configuration: {yaml_path}")
    
    # 3. Find the best weights
    model_path = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
            
    if not model_path.exists():
        print(f"❌ Could not find trained YOLO weights at {model_path}.")
        return
        
    print(f"  Loading YOLO student model: {model_path}")
    
    # 4. Run Evaluation
    print("\n🚀 Running mAP Evaluation against HUMAN GOLD STANDARD Boxes...")
    try:
        model = YOLO(str(model_path))
        
        # Suppress verbose output if possible, but we want the metrics
        metrics = model.val(data=str(yaml_path), split='val', batch=16, verbose=True)
        
        # 5. Extract and print results
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        print("\n" + "="*50)
        print("🏆 HUMAN SPATIAL AUDIT RESULTS (REAL mAP)")
        print("="*50)
        print(f"  Images Evaluated     : {len(labeled_files)}")
        print(f"  Human mAP@0.50       : {map50:.4f}")
        print(f"  Human mAP@0.50:0.95  : {map50_95:.4f}")
        print("="*50)
        
        print("\n✅ Evaluation complete. You can now insert this Real Human mAP score into your thesis!")
        
    except Exception as e:
        print(f"\n❌ Error during YOLO evaluation: {e}")

if __name__ == "__main__":
    main()
