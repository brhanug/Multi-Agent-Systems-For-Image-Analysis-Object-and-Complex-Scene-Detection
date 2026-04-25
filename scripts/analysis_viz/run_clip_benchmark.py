#!/usr/bin/env python3
import os
import torch
import open_clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
SUBSET_DIR = BASE_DIR / "human_baseline_subset"
MODEL_NAME = "ViT-B-32" # Standard baseline
PRETRAINED = "laion2b_s34b_b79k"

# 10 Archival Classes
CLASSES = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
PROMPTS = [f"a historical archival photo of a {c}" for c in CLASSES]

def run_clip_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading CLIP ({MODEL_NAME}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    text = tokenizer(PROMPTS).to(device)
    
    images = sorted([f for f in os.listdir(SUBSET_DIR) if f.endswith('.jpg')])
    results = []
    
    print(f"🔍 Running Zero-Shot Retrieval on {len(images)} images...")
    for img_name in tqdm(images):
        img_path = SUBSET_DIR / img_name
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(3)
            
            top_preds = []
            for value, index in zip(values, indices):
                top_preds.append({
                    "class": CLASSES[index],
                    "score": float(value)
                })
            
            results.append({
                "image": img_name,
                "top_1": top_preds[0]["class"],
                "top_1_score": top_preds[0]["score"],
                "top_3": [p["class"] for p in top_preds]
            })
            
    # Save results
    df = pd.DataFrame(results)
    output_path = BASE_DIR / "results/clip_cross_archive_results.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ CLIP benchmark complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_clip_benchmark()
