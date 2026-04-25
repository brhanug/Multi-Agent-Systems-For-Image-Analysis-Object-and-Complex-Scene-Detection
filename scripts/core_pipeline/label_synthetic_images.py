#!/usr/bin/env python3
import os
import torch
import yaml
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
CONFIG_PATH = BASE_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = BASE_DIR / "weights/groundingdino_swint_ogc.pth"
AUG_DATA_DIR = BASE_DIR / "data/augmented_dataset_v1"
OUTPUT_LABELS_DIR = AUG_DATA_DIR / "labels/train"
OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "weapon": 4,
    "vehicle": 5,
    "clothing": 7
}

PROMPTS = {
    "weapon": "weapon . rifle . sword . gun . cannon",
    "vehicle": "vehicle . carriage . cart . wagon . bicycle",
    "clothing": "clothing . uniform . dress . traditional costume"
}

def main():
    device = "cpu"
    print(f"🚀 Loading GroundingDINO on {device}...")
    model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))
    model = model.to(device).eval()

    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = AUG_DATA_DIR / cls_name
        if not cls_dir.exists():
            continue
            
        print(f"✨ Labeling class: {cls_name}")
        images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        prompt = PROMPTS[cls_name]

        for img_name in tqdm(images):
            img_path = cls_dir / img_name
            try:
                image_source, image = load_image(str(img_path))
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.30,
                    text_threshold=0.25,
                    device=device
                )

                # Save in YOLO format: class x_center y_center width height (all normalized)
                # GroundingDINO already returns normalized cx, cy, w, h
                label_path = OUTPUT_LABELS_DIR / f"{Path(img_name).stem}.txt"
                with open(label_path, 'w') as f:
                    for i in range(len(boxes)):
                        # Double check if the phrase matches the target class
                        phrase = phrases[i].lower()
                        # We are already in a specific folder, but we want to be safe
                        box = boxes[i].tolist()
                        f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")

    print(f"✅ Labeling complete. Labels saved to {OUTPUT_LABELS_DIR}")

if __name__ == "__main__":
    main()
