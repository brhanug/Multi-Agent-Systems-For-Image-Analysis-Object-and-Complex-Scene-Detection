import pandas as pd
from pathlib import Path
import numpy as np
import re

TAXONOMY_CLASSES = [
    "child", "portrait", "market", "street", "tree", "illustration", "window", "field", "table", "crowd",
    "book", "church", "animal", "chair", "grass", "water", "horse", "door", "fence", "building", "sky",
    "uniform", "boat", "wall", "person", "weapon", "clothing", "bicycle", "flag", "bell", "lamp", "helmet",
    "vehicle", "text", "bridge", "tower", "map"
]

def normalize_id(raw: str) -> str:
    parts = Path(str(raw)).parts
    if len(parts) >= 2:
        if parts[-2] not in ["original", "restored", "images", "metadata", "results", "CycleGAN", "pix2pix"]:
            return f"{parts[-2]}/{Path(parts[-1]).stem}"
    return Path(str(raw)).stem

def reconstruct_key_from_raw_name(raw_name: str) -> str:
    filename = Path(str(raw_name)).name
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return stem

def reconstruct_key_from_image_id_300(img_id: str) -> str:
    parts = str(img_id).split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return img_id

def class_in_text(text: str, cls: str) -> bool:
    if not text or text == "nan":
        return False
    # Allow partial match (e.g., "children" matches "child")
    pattern = re.compile(rf'\b{re.escape(cls[:4])}', re.IGNORECASE)
    return bool(pattern.search(text))

BASE = Path("/data/brhanu/thesis_project")
RESULTS_DIR = BASE / "results/multi_agent"

scores = pd.read_csv(RESULTS_DIR / "agent_comparison_scores.csv")
manifest = pd.read_csv(BASE / "final_dataset/metadata/manifest.csv")

scores["img_key"] = scores["image_id"].apply(normalize_id)
manifest["img_key"] = manifest["image_id"].apply(normalize_id)

df = scores.merge(manifest, on="img_key", how="inner", suffixes=("","_m"))
df["combined_text"] = (df["blip2_caption"].fillna("") + " " + df["scene_tags_clip"].fillna("")).str.lower()
print(f"Global dataset size: {len(df)}")

df["stem"] = df["img_key"].apply(lambda x: Path(x).stem)

# Load the 800 labeled cohort
df_800 = pd.read_csv(BASE / "human_spatial_audit/user_annotations_800.csv")
df_800["img_key"] = df_800["raw_name"].apply(reconstruct_key_from_raw_name)
keys_800 = set(df_800["img_key"].tolist())

df_800_matched = df[df["img_key"].isin(keys_800)]
print(f"Matched rows in df for 800 cohort: {len(df_800_matched)}")

# Load the 300 labeled cohort
df_300 = pd.read_csv(BASE / "human_spatial_audit/user_annotations_300.csv")
df_300["img_key"] = df_300["image_id"].apply(reconstruct_key_from_image_id_300)
keys_300 = set(df_300["img_key"].tolist())

df_300_matched = df[df["img_key"].isin(keys_300)]
print(f"Matched rows in df for 300 cohort: {len(df_300_matched)}")

# Misalignment logic: student_v3_count == 0 & vlm_agent > 0.5
def compute_misalignment(subset_df, name):
    vlm_high = subset_df["vlm_agent"] > 0.5
    student_zero = subset_df["student_v3_count"].fillna(0) == 0
    detector_fail = (student_zero & vlm_high).sum()
    n_total = len(subset_df)
    pct = 100 * detector_fail / n_total if n_total > 0 else 0
    print(f"{name} Misalignment Rate (student_v3_count == 0 & vlm_agent > 0.5): {detector_fail} / {n_total} ({pct:.2f}%)")

compute_misalignment(df, "Global (12,110)")
compute_misalignment(df_800_matched, "800 Labeled Cohort")
compute_misalignment(df_300_matched, "300 Labeled Cohort")

# Taxonomy coverage overall:
def compute_taxonomy_coverage(subset_df, name):
    class_presence = {}
    for cls in TAXONOMY_CLASSES:
        class_presence[cls] = subset_df["combined_text"].apply(lambda t: class_in_text(str(t), cls)).to_numpy(dtype=int)
    
    presence_matrix = np.stack(list(class_presence.values()), axis=1)  # (n, 37)
    n_classes_per_img = presence_matrix.sum(axis=1)
    cov_1  = int((n_classes_per_img >= 1).sum())
    n_total = len(subset_df)
    pct = 100 * cov_1 / n_total if n_total > 0 else 0
    print(f"{name} Taxonomy Coverage (>=1 class): {cov_1} / {n_total} ({pct:.2f}%)")

compute_taxonomy_coverage(df, "Global (12,110)")
compute_taxonomy_coverage(df_800_matched, "800 Labeled Cohort")
compute_taxonomy_coverage(df_300_matched, "300 Labeled Cohort")
