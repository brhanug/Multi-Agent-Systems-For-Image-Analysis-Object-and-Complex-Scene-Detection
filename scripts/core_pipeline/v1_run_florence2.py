#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

# === CONFIG ===
BASE_DIR = "/data/brhanu/thesis_project"
MODEL_ID = "microsoft/Florence-2-large"
DEVICE = "cuda:2"
INPUT_DIR = os.path.join(BASE_DIR, "final_dataset/images/restored")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_v1/florence2")
LABEL_MAPPING = {
    "man": "person", "woman": "person", "person": "person", "people": "person", "human face": "person",
    "child": "child", "boy": "child", "girl": "child", "horse": "horse",
    "building": "building", "house": "building", "tower": "building", "castle": "building",
    "weapon": "weapon", "gun": "weapon", "sword": "weapon", "cannon": "weapon",
    "vehicle": "vehicle", "car": "vehicle", "carriage": "vehicle", "boat": "vehicle", "bicycle": "vehicle",
    "tree": "tree", "plant": "tree", "forest": "tree", "clothing": "clothing", "hat": "clothing", "dress": "clothing", 
    "uniform": "clothing", "text": "text", "writing": "text", "inscription": "text",
    "animal": "animal", "dog": "animal", "bird": "animal", "cow": "animal"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# === INFERENCE ===
images = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(root, f))
images.sort()

print(f"🔥 Running Florence-2 v1.0 on {len(images)} images...")

for img_path in tqdm(images):
    rel_path = os.path.relpath(img_path, INPUT_DIR)
    img_id = rel_path.replace(os.sep, "_")
    json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_id)[0]}.json")

    if os.path.exists(json_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        task_prompt = '<OD>'
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, do_sample=False, use_cache=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        
        detections = []
        data = parsed_answer.get('<OD>', {})
        bboxes = data.get('bboxes', [])
        labels = data.get('labels', [])
        
        for box, label in zip(bboxes, labels):
            label_clean = label.lower().strip()
            mapped_label = None
            if label_clean in LABEL_MAPPING:
                mapped_label = LABEL_MAPPING[label_clean]
            else:
                for key in LABEL_MAPPING:
                    if key in label_clean:
                        mapped_label = LABEL_MAPPING[key]
                        break
            
            if mapped_label:
                x1, y1, x2, y2 = box
                detections.append({
                    "label": mapped_label, "box": [x1/w, y1/h, x2/w, y2/h],
                    "confidence": 0.9, "original_label": label_clean
                })
        
        with open(json_path, "w") as f:
            json.dump({"image": rel_path, "detections": detections}, f, indent=2)

    except Exception as e:
        print(f"⚠️ Error {img_id}: {e}")

print("✅ Florence-2 v1.0 complete.")
