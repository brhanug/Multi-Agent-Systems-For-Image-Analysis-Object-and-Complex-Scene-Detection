#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO, SAM
import yaml

# ==============================
# CONFIGURATION
# ==============================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CFG = load_config()
BASE_DIR = CFG['project']['base_dir']
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored")
# Fallback to restored if diffusion_restored is empty
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) == 0:
    IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/restored")

OUTPUT_DIR = os.path.join(BASE_DIR, "results/agent0_seg_crops")
OUTPUT_JSON = os.path.join(BASE_DIR, "results/yolo11_seg_sam_results.json")

def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg + SAM Mask Refinement")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Loading YOLOv11-seg and SAM models on {DEVICE}...")
    # Load YOLOv11-seg model
    yolo_model = YOLO("yolo11m-seg.pt")
    
    # Load SAM model
    sam_model = SAM("sam_b.pt")

    # Find images
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Image directory {IMAGE_DIR} does not exist.")
        sys.exit(1)

    all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if args.limit:
        all_images = all_images[:args.limit]
        print(f"⚠️ Limiting to {args.limit} images for testing.")

    print(f"🔥 Processing {len(all_images)} images...")
    
    results_dict = {}

    for img_name in tqdm(all_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_id = os.path.splitext(img_name)[0]
        
        try:
            # 1. Run YOLOv11-seg to detect objects
            yolo_results = yolo_model.predict(img_path, device=DEVICE, verbose=False)
            
            img_cv = cv2.imread(img_path)
            h_img, w_img, _ = img_cv.shape
            
            boxes = yolo_results[0].boxes
            
            img_detections = []
            
            if len(boxes) > 0:
                # Get boxes in [x1, y1, x2, y2] format and confs
                xyxy_list = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Filter classes of interest (e.g. COCO person=0, horse=17, vehicle=2,3,5,7, animal=15,16,18,19,20,21,22,23, knife/weapon=43)
                # We can run SAM on all detections to extract high-quality masks
                
                # 2. Run SAM prompted by YOLOv11 bounding boxes
                # Ultralytics SAM predict takes the bboxes as prompts
                sam_results = sam_model.predict(img_path, bboxes=xyxy_list, device=DEVICE, verbose=False)
                
                # Parse SAM results
                sam_masks = sam_results[0].masks
                
                if sam_masks is not None:
                    masks_data = sam_masks.xy # Coordinates of polygons
                    masks_binary = sam_masks.data.cpu().numpy() # Binary masks
                    
                    for idx, (bbox, cls_id, conf) in enumerate(zip(xyxy_list, classes, confs)):
                        x1, y1, x2, y2 = map(int, bbox)
                        class_name = yolo_model.names[int(cls_id)]
                        
                        # Extract crop using binary mask
                        mask = masks_binary[idx]
                        # Resize mask to original image size if needed
                        if mask.shape[0] != h_img or mask.shape[1] != w_img:
                            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                            
                        # Apply mask to crop
                        crop = img_cv.copy()
                        crop[mask == 0] = 0
                        # Crop to bounding box
                        crop_box = crop[y1:y2, x1:x2]
                        
                        # Save crop image
                        crop_name = f"{img_id}_obj_{idx}_{class_name}.png"
                        crop_path = os.path.join(OUTPUT_DIR, crop_name)
                        cv2.imwrite(crop_path, crop_box)
                        
                        # Polygon points for json export
                        poly_pts = []
                        if idx < len(masks_data):
                            poly_pts = masks_data[idx].tolist()
                            
                        img_detections.append({
                            "object_index": idx,
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": float(conf),
                            "bbox": [x1, y1, x2, y2],
                            "polygon": poly_pts,
                            "crop_path": os.path.relpath(crop_path, BASE_DIR)
                        })
            
            results_dict[img_id] = {
                "image_name": img_name,
                "width": w_img,
                "height": h_img,
                "detections": img_detections
            }
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"✅ Instance segmentation completed. Results saved to {OUTPUT_JSON} and crops in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
