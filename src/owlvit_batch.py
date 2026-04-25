import os
import csv
import torch
import time
import gc
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
ROOT_DIR = "/data/brhanu/thesis_project/data/colibri/images/data"
MODEL_PATH = "/data/brhanu/models/owlvit-base-patch16"
OUTPUT_CSV = "/data/brhanu/thesis_project/results/owlvit_results.csv"
LOG_DIR = "/data/brhanu/thesis_project/results/logs"

TEXT_QUERIES = [["child", "family", "book", "horse", "tree", "building", "house", "street", "animal"]]
SCORE_THRESHOLD = 0.25
LOG_INTERVAL = 50  # write progress every N images

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

# Load model locally (no internet required)
print("🔄 Loading OWL-ViT model from local path...")
processor = OwlViTProcessor.from_pretrained(MODEL_PATH)
model = OwlViTForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("✅ Model loaded successfully!")

# -------------------------------------------------
# LOAD PREVIOUS RESULTS (for auto-resume)
# -------------------------------------------------
processed_images = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        next(f, None)  # skip header
        for line in f:
            processed_images.add(line.strip().split(",")[0])
    print(f"🔁 Resuming: found {len(processed_images)} processed images.")
else:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "score", "xmin", "ymin", "xmax", "ymax"])
    print("🆕 Starting new results file.")

# -------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------
total_new = 0
start_time = time.time()

for folder, _, files in os.walk(ROOT_DIR):
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        continue

    print(f"\n📁 Processing folder: {folder} ({len(image_files)} images)")
    folder_new = 0

    for file in tqdm(image_files, desc=os.path.basename(folder), leave=False):
        image_path = os.path.join(folder, file)
        if image_path in processed_images:
            continue

        try:
            # Load and preprocess
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=TEXT_QUERIES, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process predictions
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            # Write detections to CSV
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= SCORE_THRESHOLD:
                        box = [round(i, 2) for i in box.tolist()]
                        writer.writerow([image_path, TEXT_QUERIES[0][label], float(score), *box])

            total_new += 1
            folder_new += 1
            processed_images.add(image_path)

            # Periodic GPU cleanup
            if total_new % LOG_INTERVAL == 0:
                torch.cuda.empty_cache()
                gc.collect()
                logging.info(f"Processed {total_new} images so far.")

        except Exception as e:
            logging.error(f"❌ Error processing {image_path}: {e}")
            continue

    print(f"✅ Folder done: {folder_new} new images processed.")
    logging.info(f"Folder {folder} done: {folder_new} new images.")

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
elapsed = (time.time() - start_time) / 60
summary = f"\n🏁 Completed {total_new} new images in {elapsed:.2f} minutes."
print(summary)
logging.info(summary)
