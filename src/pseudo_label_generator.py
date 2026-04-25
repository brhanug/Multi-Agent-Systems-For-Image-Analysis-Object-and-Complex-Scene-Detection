import os
import pandas as pd
from tqdm import tqdm

# Paths
csv_path = "/data/brhanu/thesis_project/results/owlvit_results.csv"
output_dir = "/data/brhanu/thesis_project/data/pseudo_labels"
os.makedirs(output_dir, exist_ok=True)

# Load detections
df = pd.read_csv(csv_path)

# Filter by confidence threshold
threshold = 0.30
df = df[df["score"] > threshold]

print(f"✅ Retained {len(df)} detections with confidence > {threshold}")

# Map labels to YOLO class IDs
label_map = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
print("Label map:", label_map)

# Group detections by image and save YOLO-format files
for img_path, group in tqdm(df.groupby("image_path")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(output_dir, f"{base}.txt")

    with open(txt_path, "w") as f:
        for _, row in group.iterrows():
            cls_id = label_map[row["label"]]
            # Convert coordinates to YOLO normalized format
            # [class, x_center, y_center, width, height]
            x_center = ((row["xmin"] + row["xmax"]) / 2.0) / 2048
            y_center = ((row["ymin"] + row["ymax"]) / 2.0) / 2048
            width = (row["xmax"] - row["xmin"]) / 2048
            height = (row["ymax"] - row["ymin"]) / 2048
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
