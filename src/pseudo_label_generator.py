"""
Pseudo-label generator: converts OWL-ViT CSV detections to YOLO-format .txt files.

Bug fix vs original: the original script hardcoded image size as 2048x2048,
producing wrong YOLO-normalized coordinates for any image that differs.
This version reads the actual width/height from each image file.
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (overridable via env vars)
# ---------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    "OWLVIT_CSV", "/data/brhanu/thesis_project/results/owlvit_results.csv"
)
IMAGE_ROOT = os.environ.get(
    "IMAGE_ROOT", "/data/brhanu/thesis_project/data/colibri/images/data"
)
OUTPUT_DIR = os.environ.get(
    "PSEUDO_LABEL_DIR", "/data/brhanu/thesis_project/data/pseudo_labels"
)
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.30"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["score"] > SCORE_THRESHOLD].copy()
print(f"Retained {len(df)} detections with confidence > {SCORE_THRESHOLD}")

label_map = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
print("Label map:", label_map)

# ---------------------------------------------------------------------------
# Cache image sizes to avoid re-opening the same file multiple times
# ---------------------------------------------------------------------------
_size_cache: dict = {}


def get_image_size(img_path: str):
    """Return (width, height) for img_path, using a module-level cache."""
    if img_path not in _size_cache:
        try:
            with Image.open(img_path) as im:
                _size_cache[img_path] = im.size  # PIL returns (width, height)
        except Exception:
            print(f"WARNING: Could not open {img_path} - skipping size lookup")
            _size_cache[img_path] = (1, 1)
    return _size_cache[img_path]


# ---------------------------------------------------------------------------
# Group by image and write YOLO labels
# ---------------------------------------------------------------------------
skipped = 0
written = 0

for img_path, group in tqdm(df.groupby("image_path"), desc="Writing labels"):
    base = Path(img_path).stem
    txt_path = os.path.join(OUTPUT_DIR, f"{base}.txt")

    img_w, img_h = get_image_size(img_path)
    if img_w <= 1 or img_h <= 1:
        skipped += 1
        continue

    with open(txt_path, "w") as f:
        for _, row in group.iterrows():
            cls_id = label_map[row["label"]]
            # Normalize by ACTUAL image dimensions (fixes hardcoded-2048 bug)
            x_center = ((row["xmin"] + row["xmax"]) / 2.0) / img_w
            y_center = ((row["ymin"] + row["ymax"]) / 2.0) / img_h
            width = (row["xmax"] - row["xmin"]) / img_w
            height = (row["ymax"] - row["ymin"]) / img_h

            # Clamp to [0, 1] to guard against slight out-of-bounds boxes
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    written += 1

print(f"Wrote {written} label files to {OUTPUT_DIR}")
if skipped:
    print(f"Skipped {skipped} images (could not determine size).")
