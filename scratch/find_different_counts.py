import pandas as pd
from pathlib import Path

base = Path("/data/brhanu/thesis_project")
manifest_path = base / "final_dataset/metadata/manifest.csv"
df = pd.read_csv(manifest_path)

diff_df = df[(df["vqa_total_objects"] != df["student_v3_count"]) & (df["student_v3_count"] > 0)]
print(f"Total images with YOLO count > 0 and VLM count != YOLO count: {len(diff_df)}")

# Let's inspect some samples
samples = diff_df.head(20)
for i, row in samples.iterrows():
    print(f"ID: {row['image_id']} | VLM: {row['vqa_total_objects']} | YOLO: {row['student_v3_count']}")
