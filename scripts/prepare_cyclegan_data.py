
import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
ENHANCED_PATH = "/data/brhanu/thesis_project/pytorch-CycleGAN-and-pix2pix/results/hist2modern_cyclegan_v2/test_latest/images"
OUTPUT_DIR = "/data/brhanu/thesis_project/data/labels/cyclegan_pseudo_v2"
YOLO_MODEL_PATH = "/data/brhanu/thesis_project/runs/detect/train_refine_long3/weights/best.pt"
CONF_THRESHOLD = 0.2  # adjust between 0.1-0.25 if too few detections

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD YOLO MODEL
# ------------------------------------------------------------
print("🔍 Loading YOLO model...")
yolo = YOLO(YOLO_MODEL_PATH)
print(f"✅ Model loaded from: {YOLO_MODEL_PATH}")

# ------------------------------------------------------------
# RUN PSEUDO-LABEL GENERATION
# ------------------------------------------------------------
print(f"📸 Scanning enhanced images from: {ENHANCED_PATH}")
images = [f for f in os.listdir(ENHANCED_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(images)} images for pseudo-labeling.\n")

detections = []

for img_name in tqdm(images, desc="Generating pseudo-labels"):
    img_path = os.path.join(ENHANCED_PATH, img_name)
    preds = yolo.predict(img_path, conf=CONF_THRESHOLD, verbose=False)

    # If no detections, skip
    if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
        continue

    # Process boxes
    for box in preds[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if conf < CONF_THRESHOLD:
            continue
        w, h = x2 - x1, y2 - y1
        x_center, y_center = x1 + w / 2, y1 + h / 2
        detections.append({
            "image": img_name,
            "class": int(cls),
            "confidence": conf,
            "x_center": x_center,
            "y_center": y_center,
            "width": w,
            "height": h
        })

        # Save as YOLO-formatted label
        label_file = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "a") as f:
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# ------------------------------------------------------------
# SAVE SUMMARY CSV
# ------------------------------------------------------------
if detections:
    df = pd.DataFrame(detections)
    csv_path = os.path.join(OUTPUT_DIR, "cyclegan_v2_pseudo_labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(detections)} detections to {csv_path}")
else:
    print("⚠️ No detections were generated — check confidence threshold or input quality.")

print(f"📂 YOLO label files saved in: {OUTPUT_DIR}")
print("✅ Pseudo-label generation completed.")