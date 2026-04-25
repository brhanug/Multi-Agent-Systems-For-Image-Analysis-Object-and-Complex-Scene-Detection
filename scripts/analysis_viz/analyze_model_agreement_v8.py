#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_model_agreement_v8.py
------------------------------------------------
Final model agreement analysis including Student Model (Iter 3):
 - Incorporates Florence-2 detections
 - Incorporates self-trained YOLO Student (v3)
 - Computes 10 pairwise agreements
 - Correlates with scene categories
------------------------------------------------
"""

import os, json, csv, re, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel

# ========= CONFIG ==========
BASE = "/data/brhanu/thesis_project/results"
PATHS = {
    "owlvit": f"{BASE}/owlvit_refined_v1/owlvit_refined.json",
    "dino": f"{BASE}/aligned_detections/groundingdino_aligned.json",
    "yolo_teacher": f"{BASE}/aligned_detections/yolo_aligned.json",
    "yolo_student": f"{BASE}/aligned_detections/student_iter_3_aligned.json",
    "florence": f"{BASE}/florence2_detections_v1.json",
    "images": f"{BASE}/dataset_export/yolo11_dataset_v3/images",
    "scenes": f"{BASE}/scene_labels/scene_labels_clip.json",
}
OUT_DIR = f"{BASE}/agreement_scores_v8"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IOU_THRESHOLD = 0.3
CLIP_MODEL = "openai/clip-vit-base-patch32"

# ========= CLIP setup ==========
print(f"🔹 Loading CLIP model ({CLIP_MODEL}) on {DEVICE}...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

# ========= Helpers ==========
def normalize_id(img_name: str) -> str:
    base = os.path.splitext(img_name)[0]
    return re.sub(r"_(fake|real|rec)_[AB]", "", base)

def clean_label(label: str) -> str:
    words = ["family", "group", "of", "soldiers", "people", "rider", "children", "two", "horse", "man", "woman", "person", "playing", "teaching", "landscape", "portrait"]
    words = sorted(words, key=len, reverse=True)
    cleaned = label.lower()
    for w in words:
        cleaned = cleaned.replace(w, f" {w} ")
    if cleaned.startswith('a'):
        if len(cleaned) > 1 and cleaned[1] != ' ':
             cleaned = 'a ' + cleaned[1:]
    return " ".join(cleaned.split())

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

@torch.no_grad()
def get_visual_embedding(image, box):
    w, h = image.size
    b = [max(0, min(1, x)) for x in box]
    crop = image.crop([b[0]*w, b[1]*h, b[2]*w, b[3]*h])
    if crop.size[0] == 0 or crop.size[1] == 0:
        return np.zeros(512)
    inputs = clip_proc(images=crop, return_tensors="pt").to(DEVICE)
    emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

@torch.no_grad()
def get_text_embedding(label):
    if not label: return np.zeros(512)
    inputs = clip_proc(text=[label], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def load_json(path):
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return {}
    data = json.load(open(path))
    det = defaultdict(list)
    if isinstance(data, dict):
        for img_id, items in data.items():
            norm_id = normalize_id(img_id)
            for d in items:
                label, box = d.get("label", "unknown"), d.get("box", [0,0,0,0])
                if "groundingdino" in path: label = clean_label(label)
                if "groundingdino" in path:
                    cx, cy, bw, bh = box
                    box = [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]
                det[norm_id].append({"label": label, "box": box, "confidence": d.get("confidence", 0.0)})
    elif isinstance(data, list):
        for d in data:
            norm_id = normalize_id(d.get("image", d.get("image_id", "unknown")))
            for det_obj in d.get("detections", []):
                label, box = det_obj.get("label", "unknown"), det_obj.get("box", [0,0,0,0])
                if "groundingdino" in path: label = clean_label(label)
                if "groundingdino" in path:
                    cx, cy, bw, bh = box
                    box = [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]
                det[norm_id].append({"label": label, "box": box, "confidence": det_obj.get("confidence", 0.0)})
    return det

# ========= Main Execution ==========
print("🔹 Loading detections...")
models = ["owlvit", "dino", "yolo_teacher", "yolo_student", "florence"]
all_dets = {m: load_json(PATHS[m]) for m in models}

scenes = json.load(open(PATHS["scenes"]))
norm_scenes = {normalize_id(k): v[0][0] for k, v in scenes.items() if v}

all_imgs = sorted(set().union(*(d.keys() for d in all_dets.values())))
print(f"✅ Loaded {len(all_imgs)} unique image IDs.")

results = []
scene_scores = defaultdict(list)

# Define pairwise combinations for 5 models
import itertools
pair_names = [f"{m1}_{m2}" for m1, m2 in itertools.combinations(models, 2)]

print(f"🔹 Computing agreement v8 (with {len(pair_names)} model pairs)...")
for img_id in tqdm(all_imgs):
    img_path = None
    for ext in [".jpg", ".png"]:
        for suffix in ["", "_fake_B", "_fake_A", "_real_A", "_rec_A"]:
            p = os.path.join(PATHS["images"], f"{img_id}{suffix}{ext}")
            if os.path.exists(p): img_path = p; break
        if img_path: break
    if not img_path: continue
    try: img = Image.open(img_path).convert("RGB")
    except: continue

    scene_type = norm_scenes.get(img_id, "unknown")
    pair_scores = {}
    
    for m1, m2 in itertools.combinations(models, 2):
        name = f"{m1}_{m2}"
        detA, detB = all_dets[m1].get(img_id, []), all_dets[m2].get(img_id, [])
        if not detA or not detB: pair_scores[name] = 0.0; continue
        
        total_scores = []
        for da in detA:
            embA_vis, embA_txt = get_visual_embedding(img, da["box"]), get_text_embedding(da["label"])
            for db in detB:
                i = iou(da["box"], db["box"])
                if i < IOU_THRESHOLD: continue
                embB_vis, embB_txt = get_visual_embedding(img, db["box"]), get_text_embedding(db["label"])
                total_scores.append(0.5*i + 0.25*cosine(embA_vis, embB_vis) + 0.25*cosine(embA_txt, embB_txt))
        pair_scores[name] = np.mean(total_scores) if total_scores else 0.0

    final_score = np.mean(list(pair_scores.values()))
    results.append({"image_id": img_id, "scene": scene_type, **pair_scores, "final_score": final_score})
    scene_scores[scene_type].append(final_score)

# ========= Save & Plot ==========
csv_path = os.path.join(OUT_DIR, "agreement_v8_final_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "scene"] + pair_names + ["final_score"])
    writer.writeheader()
    writer.writerows(results)

plt.figure(figsize=(12, 6))
scene_means = {s: np.mean(sc) for s, sc in scene_scores.items()}
sorted_scenes = sorted(scene_means.items(), key=lambda x: x[1], reverse=True)
if sorted_scenes:
    labels, values = zip(*sorted_scenes)
    sns.barplot(x=list(labels), y=list(values), palette="magma")
    plt.title("Model Agreement per Scene Category (v8 - Final)")
    plt.ylabel("Mean Agreement Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scene_agreement_barplot_v8.png"))

# Pairwise Agreement Heatmap
matrix = np.zeros((len(models), len(models)))
np.fill_diagonal(matrix, 1.0)
m_idx = {m: i for i, m in enumerate(models)}
for name, val in {name: np.mean([r[name] for r in results]) for name in pair_names}.items():
    m1, m2 = name.rsplit("_", 1) if "_" in name else (name, "") # Handle edge cases if any
    # Actually pair_names are like m1_m2. 
    # But wait, yolo_teacher_yolo_student has two underscores.
    # I should find the right split point.
    # Let's use the explicit names.
    pass

# Refined matrix filling logic
mean_pair_scores = {name: np.mean([r[name] for r in results]) for name in pair_names}
for i, m1 in enumerate(models):
    for j, m2 in enumerate(models):
        if i == j: continue
        name = f"{m1}_{m2}"
        if name in mean_pair_scores:
            matrix[i, j] = matrix[j, i] = mean_pair_scores[name]
        else:
            # Try reversed
            name_rev = f"{m2}_{m1}"
            if name_rev in mean_pair_scores:
                matrix[i, j] = matrix[j, i] = mean_pair_scores[name_rev]

plt.figure(figsize=(10, 8))
# Clean names for labels
display_names = ["OWL-ViT", "DINO", "YOLO-Teacher", "YOLO-Student", "Florence-2"]
sns.heatmap(matrix, annot=True, cmap="YlGnBu", xticklabels=display_names, yticklabels=display_names)
plt.title("Final Cross-Model Agreement Heatmap (v8)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pairwise_agreement_heatmap_v8.png"))

print(f"\n✅ Analysis v8 Complete. Results saved in: {OUT_DIR}")
print(f"📊 Global Mean Agreement (v8): {np.mean([r['final_score'] for r in results]):.4f}")
