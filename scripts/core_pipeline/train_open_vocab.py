import os
import json
import datetime
from ultralytics import YOLO

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "/data/brhanu/thesis_project"
YOLO_DATA = os.path.join(BASE_DIR, "data/yolo11_open_vocab.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "results/yolo11_final_open_vocab")
LOG_PATH = os.path.join(BASE_DIR, "results/thesis_progress_log.md")

EPOCHS = 100
BATCH = 16
IMG_SIZE = 640
MODEL = "yolo11m.pt"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# TRAIN YOLOv11 OPEN-VOCAB
# ==========================
print("🚀 Starting YOLOv11 Phase 2: Open-Vocabulary Student Training...")

# Load model
model = YOLO(MODEL)

# Train
results = model.train(
    data=YOLO_DATA,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    project=RESULTS_DIR,
    name="exp_open_vocab",
    workers=8
)
print("\n✅ YOLOv11 Open-Vocabulary training completed.")

# ==========================
# LOAD METRICS
# ==========================
metrics_path = os.path.join(RESULTS_DIR, "exp_open_vocab", "results.json")

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        results = json.load(f)
    metrics = {
        "Precision": results.get("metrics/precision(B)", "N/A"),
        "Recall": results.get("metrics/recall(B)", "N/A"),
        "mAP@50": results.get("metrics/mAP50(B)", "N/A"),
        "mAP@50-95": results.get("metrics/mAP50-95(B)", "N/A"),
    }
else:
    metrics = {"Precision": "N/A", "Recall": "N/A", "mAP@50": "N/A", "mAP@50-95": "N/A"}

# ==========================
# LOG TO MARKDOWN
# ==========================
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

log_entry = f"""
## 🧩 YOLOv11 Open-Vocabulary Student Training — {timestamp}

**Dataset:** 37 Physically Discovered Classes (via GroundingDINO NLP Analysis)  
**Training Configuration:**  
- Model: `{MODEL}`  
- Epochs: `{EPOCHS}`  
- Batch size: `{BATCH}`  
- Image size: `{IMG_SIZE}`  
- Dataset config: `{YOLO_DATA}`  
- Results directory: `{RESULTS_DIR}/exp_open_vocab`

### 📊 Performance Metrics
| Metric | Value |
|--------|--------|
| Precision | {metrics['Precision']} |
| Recall | {metrics['Recall']} |
| mAP@50 | {metrics['mAP@50']} |
| mAP@50-95 | {metrics['mAP@50-95']} |

✅ Open-Vocabulary Student Training complete. The model now natively detects the top 37 historical objects found in the Colibri archive.
"""

with open(LOG_PATH, "a") as f:
    f.write(log_entry)

print(f"\n🧾 Results logged to: {LOG_PATH}")
print(f"📁 Check full metrics at: {metrics_path}")
print("🎯 Training pipeline complete.")
