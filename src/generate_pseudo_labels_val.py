from ultralytics import YOLO
import os

# Paths
model_path = "/data/brhanu/thesis_project/runs/detect/train3/weights/best.pt"
source_path = "/data/brhanu/thesis_project/data/images/val"
output_dir = "/data/brhanu/thesis_project/pseudo_labels/val_pseudo_v3"

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# Run prediction
results = model.predict(
    source=source_path,
    conf=0.05,          # low confidence threshold
    save=True,
    save_txt=True,
    save_conf=True,
    project="/data/brhanu/thesis_project/pseudo_labels",
    name="val_pseudo_v3",
    imgsz=640
)

print("✅ Pseudo-label generation completed!")
print(f"Results saved to: {output_dir}")
