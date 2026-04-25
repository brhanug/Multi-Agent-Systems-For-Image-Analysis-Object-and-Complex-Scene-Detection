import os
import json
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def main():
    # Configuration
    WEIGHTS = "/data/brhanu/thesis_project/runs/detect/student_iter_3_train/weights/best.pt"
    IMAGES_DIR = "/data/brhanu/thesis_project/final_dataset/images/diffusion_restored"
    OUTPUT_FILE = "/data/brhanu/thesis_project/results/aligned_detections/student_iter_3_aligned.json"
    CONF_THRES = 0.25
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"🔹 Loading student model from {WEIGHTS}...")
    model = YOLO(WEIGHTS)
    
    # Taxonomy mapping (Standard 10-class)
    names = model.names
    
    results_json = {}
    
    images = list(Path(IMAGES_DIR).glob("*.jpg"))
    print(f"🔹 Running inference on {len(images)} images...")
    
    for img_path in tqdm(images):
        results = model.predict(source=str(img_path), conf=CONF_THRES, verbose=False)
        
        img_id = img_path.stem
        detections = []
        
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                b = boxes[i]
                xyxy = b.xyxy[0].cpu().numpy() # [x1, y1, x2, y2] in pixels
                # Convert to normalized [x1, y1, x2, y2] for agreement script compatibility
                w, h = r.orig_shape[1], r.orig_shape[0]
                norm_box = [
                    float(xyxy[0] / w),
                    float(xyxy[1] / h),
                    float(xyxy[2] / w),
                    float(xyxy[3] / h)
                ]
                
                detections.append({
                    "label": names[int(b.cls[0])],
                    "box": norm_box,
                    "confidence": float(b.conf[0])
                })
        
        results_json[img_id] = detections

    print(f"🔹 Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_json, f, indent=2)
        
    print("✅ Done!")

if __name__ == "__main__":
    main()
