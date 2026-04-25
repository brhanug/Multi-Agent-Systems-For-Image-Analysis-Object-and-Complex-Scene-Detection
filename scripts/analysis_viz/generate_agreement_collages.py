#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_agreement_collages.py
------------------------------------------------------------
Generates side-by-side comparison collages of detections from:
  • OWL-ViT (blue)
  • GroundingDINO (red)
  • YOLOv11 (green)
------------------------------------------------------------
Input:
  - /results/aligned_detections/
  - /results/gold_subset/low_agreement_subset.txt
Output:
  - /results/agreement_collages/
"""

import os, json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ========= CONFIG ==========
BASE = "/data/brhanu/thesis_project/results"
ALIGN_DIR = f"{BASE}/aligned_detections"
LOW_AGR = f"{BASE}/gold_subset/low_agreement_subset.txt"
OUT_DIR = f"{BASE}/agreement_collages"
IMG_DIR = f"{BASE}/dataset_export/yolo11_dataset_v3/images"

os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "owlvit": (0, 120, 255),      # Blue
    "groundingdino": (255, 0, 0), # Red
    "yolo": (0, 255, 0)           # Green
}

LABELS = {
    "owlvit": "OWL-ViT (Zero-shot)",
    "groundingdino": "GroundingDINO (Phrase-guided)",
    "yolo": "YOLOv11 (Pseudo-supervised)"
}

# ========= HELPERS ==========

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def draw_boxes(image, detections, color):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for d in detections:
        box = d.get("box") or d.get("bbox")
        if not box: continue
        x1, y1, x2, y2 = [max(0, min(1, v)) for v in box]  # clamp
        draw.rectangle([x1*w, y1*h, x2*w, y2*h], outline=color, width=3)
    return image

def create_collage(img_path, owlvit_det, dino_det, yolo_det, out_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    imgs = []
    for model, dets in zip(["owlvit", "groundingdino", "yolo"], [owlvit_det, dino_det, yolo_det]):
        im_copy = img.copy()
        im_annotated = draw_boxes(im_copy, dets, COLORS[model])
        title_bar = Image.new("RGB", (w, 50), COLORS[model])
        draw = ImageDraw.Draw(title_bar)
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
        draw.text((15, 10), LABELS[model], fill=(255,255,255), font=font)
        combined = Image.new("RGB", (w, h+50))
        combined.paste(title_bar, (0,0))
        combined.paste(im_annotated, (0,50))
        imgs.append(combined)

    # Combine horizontally
    collage_w = sum(i.width for i in imgs)
    collage_h = max(i.height for i in imgs)
    collage = Image.new("RGB", (collage_w, collage_h), (255,255,255))

    x_offset = 0
    for i in imgs:
        collage.paste(i, (x_offset, 0))
        x_offset += i.width

    collage.save(out_path)

# ========= MAIN PIPELINE ==========

print("🔹 Loading aligned detections...")
owlvit = load_json(os.path.join(ALIGN_DIR, "owlvit_aligned.json"))
dino = load_json(os.path.join(ALIGN_DIR, "groundingdino_aligned.json"))
yolo = load_json(os.path.join(ALIGN_DIR, "yolo_aligned.json"))

# Convert list to dict
def to_dict(data):
    return {entry["image"]: entry["detections"] for entry in data if "image" in entry and "detections" in entry}
owlvit, dino, yolo = map(to_dict, [owlvit, dino, yolo])

# Load low-agreement subset
low_ids = [line.strip() for line in open(LOW_AGR).read().splitlines() if line.strip()]
print(f"✅ Loaded {len(low_ids)} low-agreement image IDs.")

# Generate collages
print("🎨 Generating triptych collages...")
for img_id in tqdm(low_ids):
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path): continue

    out_path = os.path.join(OUT_DIR, f"{img_id}_triptych.jpg")
    create_collage(
        img_path,
        owlvit.get(img_id, []),
        dino.get(img_id, []),
        yolo.get(img_id, []),
        out_path
    )

print(f"\n✅ Triptych collage generation complete.")
print(f"📁 Saved to: {OUT_DIR}")