#!/usr/bin/env python3
import os, json, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path

# === CONFIG ===
MODEL_ID = "microsoft/kosmos-2.5"
DEVICE = "cuda:3"
INPUT_DIR = "/data/brhanu/thesis_project/final_dataset/images/restored"
OUT_DIR = "/data/brhanu/thesis_project/results_v1/kosmos"

os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD MODEL ===
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, trust_remote_code=True, device_map=DEVICE, torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# === INFERENCE ===
images = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(root, f))
images.sort()

print(f"🚀 Running Kosmos-2.5 on {len(images)} images...")

for img_name in tqdm(images):
    rel_path = os.path.relpath(img_name, INPUT_DIR)
    img_id = rel_path.replace(os.sep, "_")
    json_path = os.path.join(OUT_DIR, f"{Path(img_id).stem}.json")

    if os.path.exists(json_path):
        continue

    try:
        image = Image.open(img_name).convert("RGB")
        prompt = "<md>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)
        
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        with open(json_path, "w") as f:
            json.dump({"image": rel_path, "kosmos_md": generated_text}, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error {img_id}: {e}")

print("✅ Kosmos-2.5 v1.0 complete.")
