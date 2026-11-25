#!/usr/bin/env python3
"""
Compute Spatial Restoration Score (SRS) sharpness and contrast gain
between original and super-resolved/restored archival images.

Outputs:
  final_dataset/metadata/srs_scores.json
  Updates manifest.csv columns srs_score_original, srs_score_restored, srs_gain
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path("/data/brhanu/thesis_project")
DATASET_ROOT = BASE_DIR / "final_dataset"
MANIFEST_PATH = DATASET_ROOT / "metadata" / "manifest.csv"
OUTPUT_JSON_PATH = DATASET_ROOT / "metadata" / "srs_scores.json"


def compute_image_sharpness(img_path: Path) -> float:
    """Compute Laplacian variance as a measure of high-frequency sharpness/detail."""
    if not img_path.exists():
        return 0.0
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found at {MANIFEST_PATH}")
        return

    print("📥 Loading Manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    print(f"📂 Total records: {len(df)}")

    srs_records = []
    sharpness_orig_list = []
    sharpness_rest_list = []

    print("⚡ Computing measured Spatial Restoration Scores (SRS)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(row.get("image_id", ""))
        orig_rel = str(row.get("original_path", ""))
        rest_rel = str(row.get("restored_path", ""))

        orig_full = DATASET_ROOT / orig_rel if orig_rel and not Path(orig_rel).is_absolute() else Path(orig_rel)
        rest_full = DATASET_ROOT / rest_rel if rest_rel and not Path(rest_rel).is_absolute() else Path(rest_rel)

        var_orig = compute_image_sharpness(orig_full)
        var_rest = compute_image_sharpness(rest_full)

        sharpness_orig_list.append(var_orig)
        sharpness_rest_list.append(var_rest)

    # Normalize sharpness to [0, 1] range using 95th percentile clipping
    max_orig = np.percentile(sharpness_orig_list, 95) if sharpness_orig_list else 1.0
    max_rest = np.percentile(sharpness_rest_list, 95) if sharpness_rest_list else 1.0
    norm_denom = max(max_orig, max_rest, 1e-5)

    srs_orig_norm = [min(1.0, float(v / norm_denom)) for v in sharpness_orig_list]
    srs_rest_norm = [min(1.0, float(v / norm_denom)) for v in sharpness_rest_list]
    srs_gain_norm = [round(r - o, 6) for o, r in zip(srs_orig_norm, srs_rest_norm)]

    for idx, row in df.iterrows():
        img_id = str(row.get("image_id", ""))
        srs_records.append({
            "image_id": img_id,
            "image": img_id,
            "srs_score_original": round(srs_orig_norm[idx], 6),
            "srs_score_restored": round(srs_rest_norm[idx], 6),
            "srs_gain": srs_gain_norm[idx],
        })

    # Update dataframe columns
    df["srs_score_original"] = [r["srs_score_original"] for r in srs_records]
    df["srs_score_restored"] = [r["srs_score_restored"] for r in srs_records]
    df["srs_gain"] = [r["srs_gain"] for r in srs_records]

    # Save manifest CSV and srs_scores.json
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"✅ Updated {MANIFEST_PATH}")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(srs_records, f, indent=2)
    print(f"✅ Saved {OUTPUT_JSON_PATH}")

    # Summary statistics
    gains = np.array([r["srs_gain"] for r in srs_records])
    print("\n📊 Measured SRS Gain Summary:")
    print(f"   Mean Gain: {gains.mean():+.6f}")
    print(f"   Min Gain:  {gains.min():+.6f}")
    print(f"   Max Gain:  {gains.max():+.6f}")
    print(f"   Improved images (gain > 0.01): {(gains > 0.01).sum()} / {len(gains)}")


if __name__ == "__main__":
    main()
