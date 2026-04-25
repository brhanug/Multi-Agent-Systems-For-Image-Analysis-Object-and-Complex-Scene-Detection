#!/usr/bin/env python3
import os
import torch
import yaml
import json
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
CONFIG_PATH = BASE_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = BASE_DIR / "weights/groundingdino_swint_ogc.pth"
TAXONOMY_PATH = BASE_DIR / "configs/taxonomy_v2.yaml"
INPUT_DIR = BASE_DIR / "final_dataset/images/restored"
OUTPUT_DIR = BASE_DIR / "results_v1/groundingdino"
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD TAXONOMY ===
with open(TAXONOMY_PATH) as f:
    taxonomy = yaml.safe_load(f)
prompts = taxonomy['groundingdino_prompts']
full_prompt = " . ".join(prompts)
box_threshold = taxonomy['thresholds']['groundingdino']['box_threshold']
text_threshold = taxonomy['thresholds']['groundingdino']['text_threshold']

# === LOAD MODEL ===
model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))
model = model.to(DEVICE).eval()

# === INFERENCE ===
images = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(root, f))
images.sort()

print(f"📸 Running GroundingDINO on CPU fallback for {len(images)} images")

for img_path in tqdm(images):
    rel_path = os.path.relpath(img_path, INPUT_DIR)
    img_id = rel_path.replace(os.sep, "_")
    json_path = os.path.join(OUTPUT_DIR, f"{Path(img_id).stem}.json")
    
    if os.path.exists(json_path):
        continue

    try:
        _, image = load_image(img_path)
        # Move inputs to device (cpu)
        boxes, logits, phrases = predict(
            model=model, image=image, caption=full_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold, device=DEVICE
        )
        
        detections = []
        for i in range(len(phrases)):
            phrase_lower = phrases[i].lower()
            detected_class = "person"
            for obj_class in taxonomy['object_classes']:
                if obj_class in phrase_lower:
                    detected_class = obj_class
                    break
            
            detections.append({
                "label": detected_class, "confidence": float(logits[i]), "bbox": boxes[i].tolist()
            })
        
        with open(json_path, 'w') as f:
            json.dump({"image": rel_path, "detections": detections}, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Error {img_id}: {e}")

print("✅ GroundingDINO v1.0 CPU complete.")
