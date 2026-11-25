#!/usr/bin/env python3
"""
build_corrected_negative_control_pool.py
----------------------------------------
Rebuilds results/multi_agent/corrected_clean_pool.csv -- the 1,250-image
trustworthy-label pool that every headline number in this thesis now rests on
(SAA ROC-AUC 0.860, positive/negative decomposition 0.9687/0.1152, AWLF 0.8702,
and the full Conformal Risk Control result: R=0.0201, 60.6% auto-approval at
91.1% recall, 2.31-3.09x lift).

WHY THIS SCRIPT EXISTS: during a full-project provenance audit the pool file was
found to be an ORPHAN -- `build_corrected_clean_holdout_split.py` only ever reads
it via --pool-csv, and no script in the repository wrote it. It had been produced
by an ad-hoc interactive session. That is precisely the failure mode this project
has repeatedly been burned by (a cited artifact with no generator), so the build
is captured here and verified to reproduce the stored file exactly.

The pool composes three REAL label sources, and deliberately DISCARDS the
never-reviewed remainder of the 1,199-row default-labeled block rather than
assuming those images are negative:

  1. cvat_expert_review (n=801)  -- the genuine C1 expert cohort, rows 0-800 of
     human_spatial_audit/user_annotations_800.csv.
  2. single_rater_reverification_2026 (n=144) -- the first visual re-audit of the
     unverified block (results/multi_agent/negative_control_reverification_audit.json),
     which found 66.7% of those default "no scene" labels to be wrong.
  3. user_second_rater_labelstudio_2026 (n=305) -- an independent second rater's
     Label Studio pass over a disjoint slice of the same block
     (self_annotation_package/task_a_negative_controls/labeling_sheet_LS_export.csv).

Run with --verify to assert byte-identical reproduction of the committed file.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
C1_CSV = BASE / "human_spatial_audit/user_annotations_800.csv"
AUDIT_JSON = BASE / "results/multi_agent/negative_control_reverification_audit.json"
LS_EXPORT = BASE / "self_annotation_package/task_a_negative_controls/labeling_sheet_LS_export.csv"
OUT_CSV = BASE / "results/multi_agent/corrected_clean_pool.csv"
# optional fourth source: the contact-sheet pass over the first 285 of the 750 still-unreviewed rows
A2_LABELS = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
EXPANDED_CSV = BASE / "results/multi_agent/expanded/expanded_clean_pool.csv"

N_EXPERT_ROWS = 801  # rows 0..800 are the genuine CVAT-reviewed C1 cohort


def build(with_a2: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(C1_CSV)

    # full_id like "PPN1756550050_00000053_1" -> ppn_id "PPN1756550050/00000053_1"
    def split_raw_name(raw_name: str):
        stem = Path(str(raw_name)).stem              # PPN1756550050_00000053_1
        ppn, rest = stem.split("_", 1)
        return f"{ppn}/{rest}", ppn

    rows = []

    # --- source 1: genuine expert-reviewed C1 -------------------------------
    for _, r in raw.iloc[:N_EXPERT_ROWS].iterrows():
        ppn_id, ppn = split_raw_name(r["raw_name"])
        rows.append({"ppn_id": ppn_id, "ppn": ppn,
                     "has_scene": int(r["n_scene_labels"] > 0),
                     "label_source": "cvat_expert_review"})

    # --- source 2: first single-rater visual re-audit (144) -----------------
    audit = json.load(open(AUDIT_JSON))
    for rec in audit["records"]:
        full_id = rec["full_id"]                      # PPN1816495190_00000281_0
        ppn, rest = full_id.split("_", 1)
        rows.append({"ppn_id": f"{ppn}/{rest}", "ppn": ppn,
                     "has_scene": int(rec["reverified_label"] == "scene_present"),
                     "label_source": "single_rater_reverification_2026"})

    # --- source 3: independent second rater via Label Studio (305) ---------
    ls = pd.read_csv(LS_EXPORT)
    # cvat_id indexes directly into the original 2,000-row annotation table
    for _, r in ls.iterrows():
        src = raw.iloc[int(r["cvat_id"])]
        ppn_id, ppn = split_raw_name(src["raw_name"])
        rows.append({"ppn_id": ppn_id, "ppn": ppn,
                     "has_scene": int(str(r["your_label"]).strip() == "scene"),
                     "label_source": "user_second_rater_labelstudio_2026"})

    # --- source 4 (only with --with-task-a2): same rater, contact-sheet pass (285) ---
    if with_a2:
        for _, r in pd.read_csv(A2_LABELS).iterrows():
            src = raw.iloc[int(r["cvat_id"])]
            ppn_id, ppn = split_raw_name(src["raw_name"])
            assert ppn_id == r["ppn_id"], f"cvat_id {r['cvat_id']} maps to {ppn_id}, sheet says {r['ppn_id']}"
            rows.append({"ppn_id": ppn_id, "ppn": ppn, "has_scene": int(str(r["label"]).strip() == "scene"),
                         "label_source": "user_rater_contact_sheet_2026"})

    return pd.DataFrame(rows, columns=["ppn_id", "ppn", "has_scene", "label_source"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare against the committed pool instead of overwriting it")
    ap.add_argument("--with-task-a2", action="store_true",
                    help="add the 285 contact-sheet labels and write the expanded pool instead")
    args = ap.parse_args()
    global OUT_CSV
    if args.with_task_a2:
        OUT_CSV = EXPANDED_CSV

    df = build(args.with_task_a2)
    counts = df.label_source.value_counts().to_dict()
    print(f"built pool: {len(df)} rows | negatives {int((df.has_scene == 0).sum())} "
          f"| positives {int((df.has_scene == 1).sum())}")
    for k, v in counts.items():
        print(f"    {k:36s} {v}")

    if args.verify:
        old = pd.read_csv(OUT_CSV)
        same_shape = old.shape == df.shape
        # order-insensitive comparison on the identifying key
        a = old.sort_values(["label_source", "ppn_id"]).reset_index(drop=True)
        b = df.sort_values(["label_source", "ppn_id"]).reset_index(drop=True)
        identical = same_shape and a.equals(b)
        print(f"\nVERIFY vs committed {OUT_CSV.name}: "
              f"{'IDENTICAL' if identical else 'DIFFERS'}")
        if not identical:
            print(f"  committed shape {old.shape} vs rebuilt {df.shape}")
            for col in ["label_source"]:
                print(f"  committed {col}:\n{old[col].value_counts()}")
                print(f"  rebuilt  {col}:\n{df[col].value_counts()}")
            raise SystemExit(1)
    else:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
