#!/usr/bin/env python3
import os, json, requests, base64, time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# === CONFIG ===
INPUT_DIR = "/data/brhanu/thesis_project/final_dataset/images/restored"
OUT_DIR = "/data/brhanu/thesis_project/results_v1/llava"
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
CONCURRENCY = 8

# Structured questions for the "Golden Path"
QUESTIONS = [
    "Identify objects present: person, child, horse, building, weapon, vehicle, tree, clothing, text, animal. List only items found.",
    "Which scene fits best: teaching, family, playing, landscape, drawing?",
    "Describe any archival handwriting or text visible.",
    "List relational triplets in (Subject, Predicate, Object) format for the primary interactions in this image."
]

os.makedirs(OUT_DIR, exist_ok=True)

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_llava(image_path, question):
    try:
        image_b64 = encode_image_base64(image_path)
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": 256
        }
        response = requests.post(API_URL, json=payload, timeout=300)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

def process_image(img_path):
    rel_path = os.path.relpath(img_path, INPUT_DIR)
    img_id = rel_path.replace(os.sep, "_")
    json_path = os.path.join(OUT_DIR, f"{Path(img_id).stem}.json")

    if os.path.exists(json_path):
        # We check if it already has all 4 questions
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if len(data.get("responses", {})) == 4:
                    return
        except:
            pass

    responses = {}
    for q in QUESTIONS:
        responses[q] = query_llava(img_path, q)
    
    with open(json_path, "w") as f:
        json.dump({"image": rel_path, "responses": responses}, f, indent=2)

images = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(root, f))
images.sort()

print(f"🔹 Querying LLaVA v1.0 (Full Metadata) for {len(images)} images...")

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    list(tqdm(executor.map(process_image, images), total=len(images)))

print("✅ LLaVA VQA/SceneGraph v1.0 complete.")
