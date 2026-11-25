import pandas as pd
from pathlib import Path

base = Path("/data/brhanu/thesis_project")
manifest_path = base / "final_dataset/metadata/manifest.csv"
df = pd.read_csv(manifest_path)

mismatch_df = df[df["vqa_total_objects"] != df["student_v3_count"]]
print(f"Total images in manifest: {len(df)}")
print(f"Total mismatch images (vqa_total_objects != student_v3_count): {len(mismatch_df)}")

# Let's inspect some samples
samples = mismatch_df.head(20)
for i, row in samples.iterrows():
    print(f"ID: {row['image_id']} | VLM (VQA): {row['vqa_total_objects']} | YOLO (student): {row['student_v3_count']} | Caption: {row['blip2_caption']}")
