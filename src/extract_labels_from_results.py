from ultralytics import YOLO
import os

model_path = "/data/brhanu/thesis_project/runs/detect/train3/weights/best.pt"
source_path = "/data/brhanu/thesis_project/data/images/val"
output_dir = "/data/brhanu/thesis_project/pseudo_labels/val_pseudo_v4"

os.makedirs(f"{output_dir}/labels", exist_ok=True)

model = YOLO(model_path)
results = model.predict(source=source_path, conf=0.01, imgsz=640, save=False)

for result in results:
    boxes = result.boxes
    name = os.path.splitext(os.path.basename(result.path))[0]
    label_path = os.path.join(output_dir, "labels", f"{name}.txt")

    with open(label_path, "w") as f:
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x_center, y_center, w, h = box.xywh[0]
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

print("✅ Labels extracted manually and saved to:")
print(output_dir)
