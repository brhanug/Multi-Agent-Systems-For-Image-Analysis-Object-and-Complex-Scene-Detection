#!/usr/bin/env python3
"""
build_corrected_clean_holdout_split.py
-----------------------------------------
Rebuilds the group-level (publication/PPN) train/calibration/test split from
the CORRECTED clean pool (results/multi_agent/corrected_clean_pool.csv,
produced by build_corrected_negative_control_pool.py), replacing
build_clean_holdout_split.py's original 2,000-image pool.

Why this exists
----------------
build_clean_holdout_split.py's source pool (human_spatial_audit/
user_annotations_800.csv, 2,000 rows) has has_scene=0 for all 1,199 rows
beyond cvat_id 800 -- confirmed by direct code/data audit to be unverified
catalog-default placeholders, not genuine human review (see
compute_real_interannotator_kappa.py's docstring). A single-rater visual
re-verification of a random 144-image sample from that pool (seed=42) found
96/144 (66.7%, 95% CI ~59-74%) were false negatives: real scenes present,
mislabeled "no scene." Every downstream "clean," "group-level," "honest"
evaluation built on the original split (SAA disagreement decomposition, CRC
clean re-evaluation, SCI-tercile clean re-evaluation, AWLF clean holdout)
was therefore computed against ground truth that was itself wrong for a
majority of its negative-control images.

This script reruns the identical group-level splitting *procedure* -- whole
publications assigned to exactly one split, positive-rate-stratified,
60/20/20 by image count -- on the CORRECTED, trustworthy 945-image pool
(801 genuinely CVAT-reviewed C1 images + 144 single-rater re-verified pool
images; the other 1,055 never-reviewed pool images are dropped, not assumed
negative).

Caveat this split carries (disclose wherever it is cited): only 53 of 945
images (5.6%) are confirmed true negatives, so class balance is still far
from a realistic deployment population, and any metric decomposed by the
negative-control class will have a small, noisy denominator. This is an
honest, trustworthy-labels re-evaluation, not a claim of a representative
negative-control rate.

Output: results/multi_agent/corrected_clean_holdout_split.csv
        columns: ppn_id, ppn, has_scene, split, label_source
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--pool-csv", default="results/multi_agent/corrected_clean_pool.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base / args.pool_csv)
    df = df.drop_duplicates("ppn_id").reset_index(drop=True)

    print(f"Corrected master pool: {len(df)} unique trustworthy-labeled images, "
          f"{df['ppn'].nunique()} unique publications (PPNs), "
          f"positive rate {df['has_scene'].mean():.3f} "
          f"({int(df['has_scene'].sum())} positive / {int((df['has_scene']==0).sum())} negative)")

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
            deficits = {b: targets[b] - (running[b] / max(sum(running.values()), 1)) for b in bucket_names}
            bucket = max(deficits, key=deficits.get)
            split_assignment[row["ppn"]] = bucket
            running[bucket] += row["n"]

    df["split"] = df["ppn"].map(split_assignment)

    print("\nSplit composition:")
    for split_name in ["train", "calibration", "test"]:
        sub = df[df["split"] == split_name]
        n_neg = int((sub["has_scene"] == 0).sum())
        print(f"  {split_name:12s}: n={len(sub):4d} ({len(sub)/len(df)*100:5.1f}%)  "
              f"PPNs={sub['ppn'].nunique():4d}  positive_rate={sub['has_scene'].mean():.3f}  "
              f"n_negative={n_neg}")

    ppn_by_split = {s: set(df[df["split"] == s]["ppn"]) for s in ["train", "calibration", "test"]}
    assert not (ppn_by_split["train"] & ppn_by_split["calibration"])
    assert not (ppn_by_split["train"] & ppn_by_split["test"])
    assert not (ppn_by_split["calibration"] & ppn_by_split["test"])
    print("\n✅ Verified: zero publication (PPN) overlap between train/calibration/test.")

    out_csv = out_dir / "corrected_clean_holdout_split.csv"
    df[["ppn_id", "ppn", "has_scene", "split", "label_source"]].to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {out_csv}")


if __name__ == "__main__":
    main()
