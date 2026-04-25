#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_agreement_results.py
------------------------------------------------
Visual comparison of OWL-ViT, GroundingDINO, and YOLO detections.
Draws color-coded bounding boxes for interpretability.
Handles invalid or reversed bounding boxes safely.
"""

import os, json, re
from PIL import Image, ImageDraw
from tqdm import tqdm

# ======== PATHS ========
BASE = "/data/brhanu/thesis_project/results"
ALIGNED = {
    "owlvit": f"{BASE}/aligned_detections/owlvit_aligned.json",
    "dino": f"{BASE}/aligned_detections/groundingdino_aligned.json",
    "yolo": f"{BASE}/aligned_detections/yolo_aligned.json",
}
IMAGES_DIR = f"{BASE}/dataset_export/yolo11_dataset_v3/images"
OUTPUT_DIR = f"{BASE}/agreement_visuals"
LOW_AGREE_FILE = f"{BASE}/gold_subset/low_agreement_subset.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== CONFIG ========
COLORS = {
    "owlvit": (0, 150, 255),   # blue
    "dino": (255, 70, 70),     # red
    "yolo": (0, 255, 120),     # green
}
IMG_SIZE_LIMIT = (1024, 1024)

# ======== HELPERS ========

def normalize_id(name):
    return re.sub(r"_(fake|real)_[AB]", "", name)

def load_json(path):
    data = json.load(open(path))
    results = {}
    if isinstance(data, list):
        for entry in data:
            img_id = normalize_id(entry.get("image", entry.get("image_id", "")))
            results[img_id] = entry.get("detections", [])
    elif isinstance(data, dict):
        for img_id, dets in data.items():
            results[normalize_id(img_id)] = dets
    return results

# ======== LOAD DATA ========
print("🔹 Loading aligned detections...")
owlvit = load_json(ALIGNED["owlvit"])
dino = load_json(ALIGNED["dino"])
yolo = load_json(ALIGNED["yolo"])

# ======== OPTIONAL LOW-AGREEMENT FILTER ========
low_subset = []
if os.path.exists(LOW_AGREE_FILE):
    low_subset = [line.strip() for line in open(LOW_AGREE_FILE).readlines() if line.strip()]
    print(f"✅ Loaded {len(low_subset)} low-agreement image IDs.")

# ======== DRAW FUNCTION ========
def fix_box(box, w, h):
    """Ensure box coordinates are valid and within image bounds."""
    if not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    # normalize if values are between 0 and 1
    if max(x1, y1, x2, y2) <= 1:
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    # correct flipped boxes
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    # clamp to image
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def draw_boxes(image, detections, color, label_prefix):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for det in detections:
        box = det.get("box") or det.get("bbox")
        fixed = fix_box(box, w, h)
        if not fixed:
            continue
        label = det.get("label", "unknown")
        conf = det.get("confidence", 0.0)
        x1, y1, x2, y2 = fixed
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label_prefix}:{label} ({conf:.2f})"
        draw.text((x1 + 4, y1 + 4), text, fill=color)
    return image

# ======== MAIN LOOP ========
print("🎨 Generating visualizations...")
for img_id in tqdm(sorted(set(owlvit) | set(dino) | set(yolo))):
    if low_subset and img_id not in low_subset:
        continue  # visualize only low-agreement subset

    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(IMG_SIZE_LIMIT)
    except Exception:
        continue

    # Draw boxes per model
    for model, color in COLORS.items():
        if model == "owlvit":
            detections = owlvit.get(img_id, [])
        elif model == "dino":
            detections = dino.get(img_id, [])
        else:
            detections = yolo.get(img_id, [])
        img = draw_boxes(img, detections, color, model)

    out_path = os.path.join(OUTPUT_DIR, f"{img_id}_compare.jpg")
    img.save(out_path)

print(f"\n✅ Visualization complete!")
print(f"📁 Saved annotated comparisons to: {OUTPUT_DIR}")