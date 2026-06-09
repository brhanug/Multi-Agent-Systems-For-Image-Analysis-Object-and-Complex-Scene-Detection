#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# ==============================
# CONFIGURATION
# ==============================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CFG = load_config()
BASE_DIR = CFG['project']['base_dir']
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored")
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) == 0:
    IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/restored")

OUTPUT_JSON = os.path.join(BASE_DIR, "results/scene_labels/scene_labels_siglip.json")

# Scene labels corresponding to the thesis taxonomy
CANDIDATE_LABELS = [
    "drawing",
    "landscape",
    "family portrait",
    "crowd playing",
    "document or teaching building"
]

# Mapping to taxonomy keys
TAXONOMY_MAP = {
    "drawing": "drawings",
    "landscape": "landscapes",
    "family portrait": "family",
    "crowd playing": "playing",
    "document or teaching building": "teaching"
}

def main():
    parser = argparse.ArgumentParser(description="SigLIP Zero-Shot Scene Classification")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    print(f"🚀 Loading SigLIP model on {DEVICE}...")
    model_id = "google/siglip-so400m-patch14-384"
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_id).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Image directory {IMAGE_DIR} does not exist.")
        sys.exit(1)

    all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if args.limit:
        all_images = all_images[:args.limit]
        print(f"⚠️ Limiting to {args.limit} images for testing.")

    print(f"🔥 Processing {len(all_images)} images with SigLIP...")
    
    results_dict = {}

    for img_name in tqdm(all_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_id = os.path.splitext(img_name)[0]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Run inference
            inputs = processor(images=image, text=CANDIDATE_LABELS, padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                
            # SigLIP sigmoid scores
            logits = outputs.logits_per_image[0]
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            
            # Map predictions to output format
            predictions = []
            for label, prob in zip(CANDIDATE_LABELS, probs):
                taxonomy_key = TAXONOMY_MAP[label]
                predictions.append({
                    "label": taxonomy_key,
                    "score": float(prob)
                })
                
            # Sort by score descending
            predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
            results_dict[img_id] = predictions
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"✅ SigLIP scene classification completed. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
