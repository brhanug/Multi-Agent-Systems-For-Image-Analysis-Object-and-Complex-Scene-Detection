#!/usr/bin/env python3
"""
Benchmark agent outputs on gold subset annotations (when available).

Expected annotation source:
  human_baseline_gold_kit/labeling_worksheet.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OBJECT_LABEL_COLS = [
    "Person",
    "Child",
    "Horse",
    "Building",
    "Weapon",
    "Vehicle",
    "Tree",
    "Clothing",
    "Text",
    "Animal",
]


def normalize_id(v: str) -> str:
    return Path(str(v)).stem


def to_binary(v) -> int | None:
    if pd.isna(v):
        return None
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"1", "yes", "true", "y"}:
            return 1
        if lv in {"0", "no", "false", "n"}:
            return 0
        return None
    if isinstance(v, (int, float)):
        if v == 1:
            return 1
        if v == 0:
            return 0
    return None


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "n": 0}
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "n": len(y_true)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run truth-backed benchmark on gold subset.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--worksheet-csv", default="human_baseline_gold_kit/labeling_worksheet.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--existing-threshold", type=float, default=0.5)
    parser.add_argument("--scene-threshold", type=float, default=0.5)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scores_csv = Path(args.scores_csv)
    if not scores_csv.is_absolute():
        scores_csv = base / scores_csv
    worksheet_csv = Path(args.worksheet_csv)
    if not worksheet_csv.is_absolute():
        worksheet_csv = base / worksheet_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_csv)
    worksheet = pd.read_csv(worksheet_csv)
    scores["id_norm"] = scores["image_id"].astype(str).map(normalize_id)
    worksheet["id_norm"] = worksheet["Image_ID"].astype(str).map(normalize_id)
    merged = worksheet.merge(scores, on="id_norm", how="left", suffixes=("_gt", "_pred"))

    # Coverage statistics
    labeled_cells = 0
    total_cells = 0
    for col in OBJECT_LABEL_COLS:
        if col not in merged.columns:
            continue
        for v in merged[col]:
            total_cells += 1
            if to_binary(v) is not None:
                labeled_cells += 1
    scene_labeled = merged["Primary_Scene"].notna().sum() if "Primary_Scene" in merged.columns else 0

    # Existing pipeline proxy: one binary prediction for object-presence
    # We compare against each object class binary label independently as a weak proxy baseline.
    per_class_rows = []
    macro_f1_vals = []
    for col in OBJECT_LABEL_COLS:
        if col not in merged.columns:
            continue
        y_true = []
        y_pred = []
        for _, row in merged.iterrows():
            gt = to_binary(row[col])
            if gt is None:
                continue
            pred = 1 if float(row.get("existing_pipeline_agent", 0.0) or 0.0) >= args.existing_threshold else 0
            y_true.append(gt)
            y_pred.append(pred)
        m = binary_metrics(y_true, y_pred)
        per_class_rows.append({"class": col, **m})
        if m["n"] > 0:
            macro_f1_vals.append(m["f1"])

    # Scene proxy benchmark (binary "scene available")
    scene_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "n": 0}
    if "Primary_Scene" in merged.columns:
        y_true = []
        y_pred = []
        for _, row in merged.iterrows():
            gt_scene = str(row["Primary_Scene"]).strip().lower() if pd.notna(row["Primary_Scene"]) else ""
            if not gt_scene:
                continue
            y_true.append(1)
            y_pred.append(1 if float(row.get("scene_agent", 0.0) or 0.0) >= args.scene_threshold else 0)
        scene_metrics = binary_metrics(y_true, y_pred)

    class_df = pd.DataFrame(per_class_rows)
    class_path = out_dir / "gold_subset_per_class_metrics.csv"
    class_df.to_csv(class_path, index=False)

    summary = {
        "num_gold_rows": int(len(merged)),
        "object_label_completion_ratio": (labeled_cells / total_cells) if total_cells else 0.0,
        "scene_label_completion_ratio": (scene_labeled / len(merged)) if len(merged) else 0.0,
        "macro_f1_existing_proxy": (sum(macro_f1_vals) / len(macro_f1_vals)) if macro_f1_vals else 0.0,
        "scene_metrics": scene_metrics,
        "note": "Metrics depend on worksheet completeness; empty labels are skipped.",
    }

    summary_path = out_dir / "gold_subset_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Wrote {class_path}")
    print(f"✅ Wrote {summary_path}")


if __name__ == "__main__":
    main()
