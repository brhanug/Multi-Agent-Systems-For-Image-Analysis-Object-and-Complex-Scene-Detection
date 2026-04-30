"""
OWL-ViT batch inference over a directory of images.

Improvements vs original:
  1. True batch inference (configurable BATCH_SIZE) instead of one image at a time,
     giving substantial GPU throughput gains.
  2. GPU memory cleanup moved into a try/finally block so it runs even on errors.
  3. Configuration via environment variables so the script can be used without edits.
  4. Robust CSV append: flushes after every batch, not just every N images, so a
     crash mid-run loses at most one batch of results.
  5. Cleaner logging: uses Python logging module consistently instead of mixing
     print() and logging.info().
"""

import csv
import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
ROOT_DIR = os.environ.get("ROOT_DIR", "/data/brhanu/thesis_project/data/colibri/images/data")
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/brhanu/models/owlvit-base-patch16")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/data/brhanu/thesis_project/results/owlvit_results.csv")
LOG_DIR = os.environ.get("LOG_DIR", "/data/brhanu/thesis_project/results/logs")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.25"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))  # NEW: process N images at once

TEXT_QUERIES = [["child", "family", "book", "horse", "tree", "building", "house", "street", "animal"]]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"owlvit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

logger.info("Loading OWL-ViT model from %s ...", MODEL_PATH)
processor = OwlViTProcessor.from_pretrained(MODEL_PATH)
model = OwlViTForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info("Model loaded.")

# ---------------------------------------------------------------------------
# Resume support: load already-processed image paths
# ---------------------------------------------------------------------------
processed_images: set[str] = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        next(f, None)  # skip header
        for line in f:
            processed_images.add(line.strip().split(",")[0])
    logger.info("Resuming: %d images already processed.", len(processed_images))
else:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "score", "xmin", "ymin", "xmax", "ymax"])
    logger.info("Starting new results file.")

# ---------------------------------------------------------------------------
# Collect all unprocessed image paths
# ---------------------------------------------------------------------------
all_image_paths: list[str] = []
for folder, _, files in os.walk(ROOT_DIR):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(folder, fname)
            if fpath not in processed_images:
                all_image_paths.append(fpath)

logger.info("Found %d unprocessed images.", len(all_image_paths))

# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------
total_new = 0
start_time = time.time()

def process_batch(batch_paths: list[str]) -> list[dict]:
    """Run OWL-ViT on a batch of image paths; return list of detection dicts."""
    images = []
    valid_paths = []
    for p in batch_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as exc:
            logger.error("Cannot open %s: %s", p, exc)

    if not images:
        return []

    inputs = processor(
        text=[TEXT_QUERIES[0]] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img.size[1], img.size[0]] for img in images])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)

    detections = []
    for path, result in zip(valid_paths, results):
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if float(score) >= SCORE_THRESHOLD:
                box_list = [round(v, 2) for v in box.tolist()]
                detections.append({
                    "image_path": path,
                    "label": TEXT_QUERIES[0][int(label)],
                    "score": float(score),
                    "xmin": box_list[0],
                    "ymin": box_list[1],
                    "xmax": box_list[2],
                    "ymax": box_list[3],
                })
    return detections


for i in tqdm(range(0, len(all_image_paths), BATCH_SIZE), desc="Batches"):
    batch = all_image_paths[i : i + BATCH_SIZE]
    try:
        detections = process_batch(batch)
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            for det in detections:
                writer.writerow([
                    det["image_path"], det["label"], det["score"],
                    det["xmin"], det["ymin"], det["xmax"], det["ymax"],
                ])
        total_new += len(batch)
    except Exception as exc:
        logger.error("Batch %d failed: %s", i // BATCH_SIZE, exc)
    finally:
        # Always clean up GPU memory after each batch
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

elapsed = (time.time() - start_time) / 60
logger.info("Completed %d new images in %.2f minutes.", total_new, elapsed)
