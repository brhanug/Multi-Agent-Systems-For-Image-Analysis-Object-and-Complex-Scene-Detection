#!/usr/bin/env python3
"""
run_synthetic_worksheet_fill.py
--------------------------------
Auto-fills labeling_worksheet.csv from the existing synthetic_human_baseline.csv.
This generates a SYNTHETIC baseline — all labels come from LLaVA-driven majority
voting, NOT human annotation. The resulting worksheet is clearly tagged so that
run_gold_evaluation.py can warn if it is used in place of real ground truth.

Run order:
    python run_synthetic_worksheet_fill.py
Output:
    human_baseline_gold_kit/labeling_worksheet_synthetic_filled.csv   (safe copy)
    human_baseline_gold_kit/labeling_worksheet.csv                    (filled in-place)

IMPORTANT: This fills the worksheet so that E2/E4 (run_gold_evaluation.py) can
run immediately. Replace with real human labels at any point — the evaluation
scripts only read from labeling_worksheet.csv and are blind to how it was filled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
WORKSHEET = BASE / "human_baseline_gold_kit" / "labeling_worksheet.csv"
SYNTHETIC = BASE / "results" / "multi_agent" / "synthetic_human_baseline.csv"
OUTPUT_SAFE = BASE / "human_baseline_gold_kit" / "labeling_worksheet_synthetic_filled.csv"

# Map worksheet columns to synthetic CSV column names
# Worksheet has 10 class columns; synthetic has synthetic_{Class} columns
CLASS_MAP = {
    "Person":    "synthetic_Person",
    "Child":     "synthetic_Child",
    "Horse":     "synthetic_Horse",
    "Building":  "synthetic_Building",
    "Weapon":    "synthetic_Weapon",
    "Vehicle":   "synthetic_Vehicle",
    "Tree":      "synthetic_Tree",
    "Clothing":  "synthetic_Clothing",
    "Text":      "synthetic_Text",
    "Animal":    "synthetic_Animal",
}
SCENE_COL_SYNTHETIC = "vqa_primary_scene"

# ---------------------------------------------------------------------------


def normalize(img_id: str) -> str:
    """Strip PPN prefix and path separators to get bare stem."""
    return Path(str(img_id)).stem


def load_synthetic_majority(path: Path) -> pd.DataFrame:
    """
    The synthetic CSV has many rows per image_id (one per simulated annotator).
    Reduce to one row per image by majority vote across all binary columns.
    Returns DataFrame indexed by bare image stem.
    """
    print(f"📂 Loading synthetic baseline: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Normalise image key
    df["_key"] = df["image_id"].astype(str).apply(normalize)

    agg: dict[str, str] = {"_key": "first"}
    for col in CLASS_MAP.values():
        if col in df.columns:
            agg[col] = "mean"  # mean → majority when threshold at 0.5
    if SCENE_COL_SYNTHETIC in df.columns:
        agg[SCENE_COL_SYNTHETIC] = lambda s: s.mode().iloc[0] if not s.empty else ""

    grouped = df.groupby("_key").agg(agg).reset_index(drop=True)

    # Binarise class columns
    for col in CLASS_MAP.values():
        if col in grouped.columns:
            grouped[col] = (grouped[col] >= 0.5).astype(int)

    grouped = grouped.set_index("_key")
    print(f"   → {len(grouped)} unique images in synthetic baseline")
    return grouped


def main() -> None:
    if not WORKSHEET.exists():
        print(f"❌ Worksheet not found: {WORKSHEET}", file=sys.stderr)
        sys.exit(1)
    if not SYNTHETIC.exists():
        print(f"❌ Synthetic baseline not found: {SYNTHETIC}", file=sys.stderr)
        sys.exit(1)

    ws = pd.read_csv(WORKSHEET)
    synthetic = load_synthetic_majority(SYNTHETIC)

    hit = miss = 0
    filled_class = {c: 0 for c in CLASS_MAP}
    rows: list[dict] = []

    for _, row in ws.iterrows():
        # Worksheet Image_ID format: PPN.../00000047_1
        img_key = normalize(str(row["Image_ID"]).split("/")[-1]) if "/" in str(row["Image_ID"]) \
            else normalize(str(row["Image_ID"]))

        new_row = row.copy()
        new_row["Ambiguity_Notes"] = "[SYNTHETIC — replace with human labels]"

        if img_key in synthetic.index:
            hit += 1
            syn = synthetic.loc[img_key]
            for ws_col, syn_col in CLASS_MAP.items():
                if syn_col in syn.index:
                    val = int(syn[syn_col])
                    new_row[ws_col] = val
                    if val == 1:
                        filled_class[ws_col] += 1
            if SCENE_COL_SYNTHETIC in syn.index:
                new_row["Primary_Scene"] = str(syn[SCENE_COL_SYNTHETIC])
        else:
            miss += 1
            # Fallback: fill blanks with 0 so worksheet is complete
            for ws_col in CLASS_MAP:
                new_row[ws_col] = 0
            new_row["Primary_Scene"] = "unknown"
            new_row["Ambiguity_Notes"] = "[SYNTHETIC — no match, defaulted to 0]"

        rows.append(new_row)

    result = pd.DataFrame(rows)
    result.to_csv(WORKSHEET, index=False)
    result.to_csv(OUTPUT_SAFE, index=False)

    print(f"\n✅ Worksheet filled: {WORKSHEET}")
    print(f"✅ Safe copy saved:  {OUTPUT_SAFE}")
    print(f"\n📊 Coverage: {hit}/{len(ws)} matched ({hit/len(ws)*100:.1f}%)")
    print(f"   Unmatched (defaulted to 0): {miss}")
    print(f"\n⚠️  CLASS PREVALENCE (synthetic counts out of {len(ws)}):")
    for cls, cnt in filled_class.items():
        print(f"   {cls:12s}: {cnt:4d} positives ({cnt/len(ws)*100:.1f}%)")
    print("\n🔴 REMINDER: These are SYNTHETIC labels from LLaVA majority vote.")
    print("   Replace with real human annotations before final thesis submission.")


if __name__ == "__main__":
    main()
