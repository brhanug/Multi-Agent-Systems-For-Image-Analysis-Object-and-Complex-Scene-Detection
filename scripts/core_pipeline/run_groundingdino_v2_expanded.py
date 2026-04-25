#!/usr/bin/env python3
"""
run_groundingdino_v2_expanded.py
----------------------------------
GroundingDINO inference with EXPANDED 10-class taxonomy
for Professor Mandl's requirements.

New classes: person, child, horse, building, weapon, vehicle, tree, clothing, text, animal
"""

import os
import torch
import yaml
import json
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict, annotate
from tqdm import tqdm
import cv2

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
CONFIG_PATH = BASE_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = BASE_DIR / "weights/groundingdino_swint_ogc.pth"
TAXONOMY_PATH = BASE_DIR / "configs/taxonomy_v2.yaml"

# Input/Output
INPUT_DIR = BASE_DIR / "results/diffusion_restored"  # Use best quality images
OUTPUT_DIR = BASE_DIR / "results/groundingdino_v2_expanded"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD TAXONOMY ===
print("📋 Loading expanded taxonomy...")
with open(TAXONOMY_PATH) as f:
    taxonomy = yaml.safe_load(f)

# Build comprehensive prompt from taxonomy
prompts = taxonomy['groundingdino_prompts']
full_prompt = " . ".join(prompts)
print(f"🔍 Prompt: {full_prompt[:200]}...")

# Thresholds
box_threshold = taxonomy['thresholds']['groundingdino']['box_threshold']
text_threshold = taxonomy['thresholds']['groundingdino']['text_threshold']

# === LOAD MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")
print("🔍 Loading GroundingDINO model...")
model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))
model = model.to(device).eval()

# === INFERENCE ===
images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"📸 Found {len(images)} images")

all_detections = {}
stats = {cls: 0 for cls in taxonomy['object_classes']}

for img_name in tqdm(images, desc="Running GroundingDINO"):
    img_path = INPUT_DIR / img_name
    
    try:
        image_source, image = load_image(str(img_path))
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=full_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        
        # Parse detections
        detections = []
        for i in range(len(phrases)):
            # Map phrase to object class
            phrase_lower = phrases[i].lower()
            detected_class = None
            
            # Simple keyword matching (can be improved with NLP)
            for obj_class in taxonomy['object_classes']:
                if obj_class in phrase_lower:
                    detected_class = obj_class
                    break
            
            if not detected_class:
                # Fallback: use first word of phrase
                detected_class = phrases[i].split()[0]
            
            detections.append({
                "label": detected_class,
                "raw_phrase": phrases[i],
                "confidence": float(logits[i]),
                "bbox": boxes[i].tolist()
            })
            
            # Update stats
            if detected_class in stats:
                stats[detected_class] += 1
        
        all_detections[img_name] = detections
        
        # Save annotated image
        if len(boxes) > 0:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(str(OUTPUT_DIR / img_name), annotated_frame)
        
        # Save JSON
        json_path = OUTPUT_DIR / f"{Path(img_name).stem}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "image": img_name,
                "detections": detections,
                "num_detections": len(detections)
            }, f, indent=2)
    
    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")
        continue

# === SAVE SUMMARY ===
summary = {
    "total_images": len(images),
    "total_detections": sum(len(d) for d in all_detections.values()),
    "class_distribution": stats,
    "taxonomy_version": "v2.0",
    "object_classes": taxonomy['object_classes'],
    "scene_classes": taxonomy['scene_classes']
}

summary_path = OUTPUT_DIR / "detection_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Completed inference on {len(all_detections)} images")
print(f"📊 Total detections: {summary['total_detections']}")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print("\n📈 Class Distribution:")
for cls, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {cls}: {count}")
