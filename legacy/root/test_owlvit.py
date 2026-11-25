import os
import csv
import torch
import time
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
ROOT_DIR = "/data/brhanu/thesis_project/data/colibri/images/data"
OUTPUT_CSV = "/data/brhanu/thesis_project/results/owlvit_results.csv"
LOG_DIR = "/data/brhanu/thesis_project/results/logs"
TEXT_QUERIES = [["child", "family", "book", "horse", "tree", "building", "house", "street", "animal"]]
SCORE_THRESHOLD = 0.25

# -------------------------------------------------
# SETUP
# -------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"owlvit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
logging.info(f"Using device: {device}")

# Load model
print("Loading OWL-ViT model...")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")

# -------------------------------------------------
# LOAD PREVIOUS RESULTS (to skip processed images)
# -------------------------------------------------
processed_images = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        next(f)  # skip header
        for line in f:
            processed_images.add(line.strip().split(",")[0])
    print(f"🔁 Found {len(processed_images)} previously processed images.")
else:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "score", "xmin", "ymin", "xmax", "ymax"])

# -------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------
total_processed = 0
start_time = time.time()

for root, _, files in os.walk(ROOT_DIR):
    for file in tqdm(files, desc=f"Scanning {root}", leave=False):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        image_path = os.path.join(root, file)
        if image_path in processed_images:
            continue  # skip already done

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=TEXT_QUERIES, images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]

            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= SCORE_THRESHOLD:
                        box = [round(i, 2) for i in box.tolist()]
                        writer.writerow([image_path, TEXT_QUERIES[0][label], float(score), *box])

            total_processed += 1
            logging.info(f"Processed: {image_path}")

        except Exception as e:
            logging.error(f"❌ Error processing {image_path}: {e}")
            continue

elapsed = (time.time() - start_time) / 60
print(f"✅ Completed {total_processed} new images in {elapsed:.2f} minutes.")
logging.info(f"Completed {total_processed} new images in {elapsed:.2f} minutes.")
