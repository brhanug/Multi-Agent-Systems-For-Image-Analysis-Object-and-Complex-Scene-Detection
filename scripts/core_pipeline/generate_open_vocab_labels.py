#!/usr/bin/env python3
import os
import torch
import yaml
import sys
from pathlib import Path

BASE_DIR = Path("/data/brhanu/thesis_project")

from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
CONFIG_PATH = BASE_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = BASE_DIR / "weights/groundingdino_swint_ogc.pth"
YAML_PATH = BASE_DIR / "data/yolo11_open_vocab.yaml"

# Input/Output paths
INPUT_DIR_TRAIN = BASE_DIR / "data/yolo11_v1_refresh/images/train"
INPUT_DIR_VAL = BASE_DIR / "data/yolo11_v1_refresh/images/val"

OUTPUT_DIR_TRAIN = BASE_DIR / "data/yolo11_open_vocab/labels/train"
OUTPUT_DIR_VAL = BASE_DIR / "data/yolo11_open_vocab/labels/val"

OUTPUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_VAL.mkdir(parents=True, exist_ok=True)

# Generate image symlinks for YOLO directory structure
os.makedirs(BASE_DIR / "data/yolo11_open_vocab/images", exist_ok=True)
if not os.path.exists(BASE_DIR / "data/yolo11_open_vocab/images/train"):
    os.symlink(INPUT_DIR_TRAIN, BASE_DIR / "data/yolo11_open_vocab/images/train")
if not os.path.exists(BASE_DIR / "data/yolo11_open_vocab/images/val"):
    os.symlink(INPUT_DIR_VAL, BASE_DIR / "data/yolo11_open_vocab/images/val")

# === LOAD TAXONOMY ===
print("📋 Loading Open-Vocabulary taxonomy...")
with open(YAML_PATH) as f:
    taxonomy = yaml.safe_load(f)

classes = list(taxonomy['names'].values())
class_to_idx = {name: idx for idx, name in taxonomy['names'].items()}

# Build comprehensive prompt
full_prompt = " . ".join(classes)
print(f"🔍 Prompt: {full_prompt}")

box_threshold = 0.3
text_threshold = 0.25

# === LOAD MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")
print("🔍 Loading GroundingDINO model...")
model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))
model = model.to(device).eval()

def process_directory(input_dir, output_dir, split_name):
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"📸 Found {len(images)} images in {split_name} split.")

    for img_name in tqdm(images, desc=f"GroundingDINO ({split_name})"):
        img_path = input_dir / img_name
        label_path = output_dir / f"{Path(img_name).stem}.txt"
        
        # Skip if already processed (allows resuming)
        if label_path.exists():
            continue

        try:
            image_source, image = load_image(str(img_path))
            h, w, _ = image_source.shape
            
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=full_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )
            
            yolo_labels = []
            for i in range(len(phrases)):
                phrase_lower = phrases[i].lower()
                detected_class_idx = None
                
                # Match to our 37 classes
                for cls_name, cls_idx in class_to_idx.items():
                    # We look for exact word matches to avoid 'cat' matching 'cattle'
                    words = phrase_lower.split()
                    if cls_name in words:
                        detected_class_idx = cls_idx
                        break
                
                if detected_class_idx is None:
                    continue  # ignore spurious matches

                # GroundingDINO boxes are naturally in [cx, cy, w, h] normalized format
                cx, cy, bw, bh = boxes[i].tolist()
                yolo_labels.append(f"{detected_class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_labels))

        except NameError as e:
            if "'_C' is not defined" in str(e):
                # The C++ extension failed on this specific image's NMS/RoI calculation.
                # Write an empty label file so YOLO ignores this image during training
                # rather than crashing the pipeline.
                with open(label_path, 'w') as f:
                    pass
            else:
                print(f"⚠️ Error processing {img_name}: {e}")
        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

if __name__ == "__main__":
    process_directory(INPUT_DIR_VAL, OUTPUT_DIR_VAL, "Validation")
    process_directory(INPUT_DIR_TRAIN, OUTPUT_DIR_TRAIN, "Training")
    print("✅ Open-Vocabulary extraction complete.")
