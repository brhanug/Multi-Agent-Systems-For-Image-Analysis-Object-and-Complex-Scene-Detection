import os
from ultralytics import YOLO

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "/data/brhanu/thesis_project"
LAST_WEIGHTS = os.path.join(BASE_DIR, "results/yolo11_final_open_vocab/exp_open_vocab3/weights/last.pt")

if not os.path.exists(LAST_WEIGHTS):
    print(f"❌ Error: {LAST_WEIGHTS} not found!")
    exit(1)

# Load model and resume
print(f"🚀 Resuming YOLOv11 training from {LAST_WEIGHTS}...")
model = YOLO(LAST_WEIGHTS)

# Resume training
# The training parameters (epochs, results dir, etc.) are stored in the checkpoint
results = model.train(resume=True)

print("\n✅ YOLOv11 Open-Vocabulary training RESUME completed.")
