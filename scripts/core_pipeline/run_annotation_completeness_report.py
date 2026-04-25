#!/usr/bin/env python3
"""
Report annotation completeness for gold worksheet labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OBJECT_COLS = [
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


def is_filled(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"0", "1", "yes", "no", "true", "false", "y", "n"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotation completeness report.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--worksheet-csv", default="human_baseline_gold_kit/labeling_worksheet.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    worksheet = Path(args.worksheet_csv)
    if not worksheet.is_absolute():
        worksheet = base / worksheet
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(worksheet)
    total_rows = len(df)

    rows = []
    for col in OBJECT_COLS + ["Primary_Scene"]:
        if col not in df.columns:
            continue
        filled = int(df[col].apply(is_filled if col != "Primary_Scene" else lambda v: not pd.isna(v) and str(v).strip() != "").sum())
        pct = (filled / total_rows) if total_rows else 0.0
        rows.append({"field": col, "filled_count": filled, "total_rows": total_rows, "completion_ratio": pct})

    report_df = pd.DataFrame(rows).sort_values("completion_ratio")
    out_csv = out_dir / "annotation_completeness_report.csv"
    report_df.to_csv(out_csv, index=False)

    object_rows = [r for r in rows if r["field"] in OBJECT_COLS]
    summary = {
        "total_rows": total_rows,
        "macro_object_completion": (sum(r["completion_ratio"] for r in object_rows) / len(object_rows)) if object_rows else 0.0,
        "scene_completion": next((r["completion_ratio"] for r in rows if r["field"] == "Primary_Scene"), 0.0),
        "fully_unlabeled_rows": int(sum((df[OBJECT_COLS].isna().all(axis=1)) if set(OBJECT_COLS).issubset(df.columns) else [0] * total_rows)),
    }
    out_json = out_dir / "annotation_completeness_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Wrote {out_csv}")
    print(f"✅ Wrote {out_json}")


if __name__ == "__main__":
    main()
