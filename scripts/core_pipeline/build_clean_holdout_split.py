#!/usr/bin/env python3
"""
build_clean_holdout_split.py
-----------------------------
Constructs a single, group-level (publication/PPN) train/calibration/test
split over ALL available real human-annotated images, and freezes it to disk
so every downstream metric script can reuse the exact same partition.

Why this exists
----------------
Prior evaluation reused the same 801-image "C1" cohort for weight selection,
calibration, AWLF training, and headline reporting, with no reserved final
test set. A supposedly "out-of-sample" comparison (C1 -> C2) turned out to
not be independent either: C2 (n=300, spatial audit) is entirely a nested
subsample of C1's full 2,000-row pool (human_spatial_audit/user_annotations_800.csv),
not a separate cohort - every one of its 300 images already appears in that
2,000-row pool.

This script instead:
  1. Loads the full 2,000-row human-reviewed pool (superset containing the
     801-image "expert reviewed" subset and the 1,199 negative controls).
  2. Groups by publication (PPN) so that no publication's images can appear
     in more than one split (avoids within-publication leakage from shared
     photographer/era/scan-batch visual style).
  3. Greedily assigns whole PPN groups to train/calibration/test targeting
     60/20/20 by image count, balancing positive-rate (has_scene) across
     splits so no partition is degenerate.
  4. Writes the frozen split assignment to
     results/multi_agent/clean_holdout_split.csv
     so every metric (SAA, MC Dropout, entropy, AWLF, HITL routing) can be
     recomputed against the exact same train/cal/test partition.

Output columns: ppn_id, ppn, has_scene, split (train/calibration/test)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def extract_ppn_id(raw_name: str) -> tuple[str, str]:
    name = str(raw_name).replace("images/", "").rsplit(".", 1)[0]
    parts = name.split("_")
    return parts[0], "_".join(parts[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--batch1-csv", default="human_spatial_audit/user_annotations_800.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base / args.batch1_csv)
    df[["ppn", "img"]] = df["raw_name"].apply(lambda x: pd.Series(extract_ppn_id(x)))
    df["ppn_id"] = df["ppn"] + "/" + df["img"]
    df["has_scene"] = (df["n_scene_labels"] > 0).astype(int)
    df = df.drop_duplicates("ppn_id").reset_index(drop=True)

    print(f"Master pool: {len(df)} unique human-reviewed images, "
          f"{df['ppn'].nunique()} unique publications (PPNs), "
          f"positive rate {df['has_scene'].mean():.3f}")

    # ── Group-level (PPN) stratified split ──────────────────────────────────
    # Whole publications go entirely into one split (never split a PPN across
    # train/cal/test) to avoid within-publication leakage. To keep positive
    # rate comparable across splits despite that constraint, PPN groups are
    # first stratified into 5 positive-rate bins, then each bin's groups are
    # independently round-robin-allocated to train/cal/test in the target
    # 60/20/20 proportion - so no split is starved of positives or negatives.
    ppn_groups = df.groupby("ppn").agg(
        n=("ppn_id", "count"),
        pos_rate=("has_scene", "mean"),
    ).reset_index()
    ppn_groups = ppn_groups.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_bins = 5
    ppn_groups["pos_bin"] = pd.qcut(ppn_groups["pos_rate"], n_bins, labels=False, duplicates="drop")

    test_frac = 1.0 - args.train_frac - args.cal_frac
    targets = {"train": args.train_frac, "calibration": args.cal_frac, "test": test_frac}
    bucket_names = list(targets.keys())

    split_assignment: dict[str, str] = {}
    running = {"train": 0, "calibration": 0, "test": 0}
    for _, bin_df in ppn_groups.groupby("pos_bin"):
        bin_df = bin_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        for _, row in bin_df.iterrows():
            # Assign to whichever bucket is currently furthest below its target share
            deficits = {b: targets[b] - (running[b] / max(sum(running.values()), 1)) for b in bucket_names}
            bucket = max(deficits, key=deficits.get)
            split_assignment[row["ppn"]] = bucket
            running[bucket] += row["n"]

    df["split"] = df["ppn"].map(split_assignment)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("\nSplit composition:")
    for split_name in ["train", "calibration", "test"]:
        sub = df[df["split"] == split_name]
        print(f"  {split_name:12s}: n={len(sub):5d} ({len(sub)/len(df)*100:5.1f}%)  "
              f"PPNs={sub['ppn'].nunique():4d}  positive_rate={sub['has_scene'].mean():.3f}")

    # Hard verification: zero publication overlap across splits
    ppn_by_split = {s: set(df[df["split"] == s]["ppn"]) for s in ["train", "calibration", "test"]}
    overlap_tc = ppn_by_split["train"] & ppn_by_split["calibration"]
    overlap_tt = ppn_by_split["train"] & ppn_by_split["test"]
    overlap_ct = ppn_by_split["calibration"] & ppn_by_split["test"]
    assert not overlap_tc, f"LEAK: {len(overlap_tc)} PPNs in both train and calibration"
    assert not overlap_tt, f"LEAK: {len(overlap_tt)} PPNs in both train and test"
    assert not overlap_ct, f"LEAK: {len(overlap_ct)} PPNs in both calibration and test"
    print("\n✅ Verified: zero publication (PPN) overlap between train/calibration/test.")

    # Also verify zero exact-image overlap (stronger, redundant check)
    id_by_split = {s: set(df[df["split"] == s]["ppn_id"]) for s in ["train", "calibration", "test"]}
    assert not (id_by_split["train"] & id_by_split["test"]), "LEAK: exact image overlap train/test"
    assert not (id_by_split["train"] & id_by_split["calibration"]), "LEAK: exact image overlap train/cal"
    assert not (id_by_split["calibration"] & id_by_split["test"]), "LEAK: exact image overlap cal/test"
    print("✅ Verified: zero exact-image overlap between train/calibration/test.")

    out_csv = out_dir / "clean_holdout_split.csv"
    df[["ppn_id", "ppn", "has_scene", "split"]].to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {out_csv}")


if __name__ == "__main__":
    main()
