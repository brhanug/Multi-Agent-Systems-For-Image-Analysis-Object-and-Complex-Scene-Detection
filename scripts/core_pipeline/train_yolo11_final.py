import os
import json
import datetime
import subprocess
import yaml

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "/data/brhanu/thesis_project"
YOLO_DATA = os.path.join(BASE_DIR, "data/yolo11_v1_refresh.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "results/yolo11_final_v1_refresh")
LOG_PATH = os.path.join(BASE_DIR, "results/thesis_progress_log.md")

EPOCHS = 100
BATCH = 16
IMG_SIZE = 640
MODEL = "yolo11m.pt"  # can switch to 'yolo11s.pt' or 'yolo11l.pt' later

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# TRAIN YOLOv11
# ==========================
from ultralytics import YOLO
# ==========================
# CONSENSUS-FILTERED DISTILLATION (CONTRASTIVE PROXY)
# ==========================
# Following Xia et al. (2025) DOtA methodology, we proxy Label-Internal Contrastive Learning (LICL).
# By aggressively filtering the dataset with the Scene-Aware Agreement (SAA) metric,
# we discard disputed boundary boxes. The YOLOv11 student model is thus trained exclusively
# on high-confidence, contrastively isolated class representations (SAA > 0.45).
# This prevents the student from learning single-model pseudo-labeling noise.

# ==========================
# TRAIN YOLOv11
# ==========================
print("🚀 Starting YOLOv11 fine-tuning...")
print("🧪 Active Mode: Consensus-Filtered Distillation (SAA > 0.45 Proxy for LICL)")

# Load model
model = YOLO(MODEL)

# Train
results = model.train(
    data=YOLO_DATA,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    project=RESULTS_DIR,
    name="exp_final"
)
print("\n✅ YOLOv11 training completed.")

# ==========================
# LOAD METRICS
# ==========================
metrics_path = os.path.join(RESULTS_DIR, "exp_final", "results.json")

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        results = json.load(f)
    metrics = {
        "Precision": results.get("metrics/precision(B)", "N/A"),
        "Recall": results.get("metrics/recall(B)", "N/A"),
        "mAP@50": results.get("metrics/mAP50(B)", "N/A"),
        "mAP@50-95": results.get("metrics/mAP50-95(B)", "N/A"),
        "Box Loss": results.get("train/box_loss", "N/A"),
        "Class Loss": results.get("train/cls_loss", "N/A"),
        "Objectness Loss": results.get("train/dfl_loss", "N/A"),
    }
else:
    metrics = {"Precision": "N/A", "Recall": "N/A", "mAP@50": "N/A", "mAP@50-95": "N/A"}

# ==========================
# LOG TO MARKDOWN
# ==========================
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

log_entry = f"""
## 🧩 YOLOv11 Final Training Summary — {timestamp}

**Dataset:** diffusion_restored (pseudo-labeled via OWL-ViT + BLIP-2)  
**Training Configuration:**  
- Model: `{MODEL}`  
- Epochs: `{EPOCHS}`  
- Batch size: `{BATCH}`  
- Image size: `{IMG_SIZE}`  
- Dataset config: `{YOLO_DATA}`  
- Results directory: `{RESULTS_DIR}/exp_final`

### 📊 Performance Metrics
| Metric | Value |
|--------|--------|
| Precision | {metrics['Precision']} |
| Recall | {metrics['Recall']} |
| mAP@50 | {metrics['mAP@50']} |
| mAP@50-95 | {metrics['mAP@50-95']} |
| Box Loss | {metrics['Box Loss']} |
| Class Loss | {metrics['Class Loss']} |
| Objectness Loss | {metrics['Objectness Loss']} |

✅ Training complete. YOLOv11 is now aligned with BLIP-2 + OWL-ViT pseudo-labels on denoised, diffusion-restored images.  
"""

# Append to Markdown progress log
with open(LOG_PATH, "a") as f:
    f.write(log_entry)

print(f"\n🧾 Results logged to: {LOG_PATH}")
print(f"📁 Check full metrics at: {metrics_path}")
print("🎯 Training pipeline complete — ready for evaluation and visualization.")