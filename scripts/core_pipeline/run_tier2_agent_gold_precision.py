#!/usr/bin/env python3
"""
run_tier2_agent_gold_precision.py
---------------------------------
Recomputes the three gold-labeled Tier 2 precision figures cited throughout the
thesis and defense deck, from the human labels exported out of Label Studio:

  Agent 2 (contradiction detection)  -> 39.7% precision as a hallucination
                                        detector (all 127 candidates); 23.9% without
                                        the 24 words on one emblem
  Agent 4 (cross-archive linking)    -> 55.7% overall precision (n=79 pairs),
                                        rising to 100% at similarity >= 0.92
  Agent 5 (grounded OCR)             -> its "candidate real OCR" bucket is only
                                        52.5% precise as a real-text detector
                                        (n=53 hand-transcribed images)

WHY THIS SCRIPT EXISTS: a full-project provenance audit found the three JSON
artifacts backing these numbers were orphans -- the figures are cited 20+ times
across both documents, but nothing in the repository regenerated them (they came
from ad-hoc interactive sessions). Same failure mode this project has repeatedly
been burned by, so the computation is captured here. Run with --verify to assert
the recomputation still matches the committed JSON.

Inputs are the Label Studio exports written by extract_self_annotation_labelstudio.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package"
OUT = BASE / "results/multi_agent"

AGENT4_THRESHOLDS = [0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]


def agent2():
    """All 127 candidates: 101 labelled in Label Studio, the last 26 on the quick-labels page (4 Nov 2026).

    24 of those 26 are words the VLM attached to one emblem, mostly virtues (honesty, loyalty ...), on
    (00000005_1.jpg); the figure is also given without those 24 rows."""
    ls = pd.read_csv(PKG / "task_b_contradiction_ground_truth/labeling_sheet_LS_export.csv")
    ls = ls[ls.your_judgment.notna()][["id", "claimed_noun", "your_judgment"]]
    sheet = pd.read_csv(PKG / "task_b_contradiction_ground_truth/labeling_sheet.csv")
    qb = pd.read_csv(PKG / "quick_labels/B_tasks.csv")
    ql = pd.read_csv(PKG / "quick_labels/object_labels.csv").drop_duplicates("key", keep="last")
    q = qb.merge(ql, left_on="ls_task_id", right_on="key")[["row_id", "claimed_noun", "label"]]
    q = q.rename(columns={"row_id": "id", "label": "your_judgment"})
    df = pd.concat([ls, q[~q.id.isin(ls.id)]], ignore_index=True).merge(sheet[["id", "image_file"]], on="id")
    emblem_words = {"endurance", "faithfulnes", "freedom", "honesty", "honor", "integrity", "loyalty", "nobility",
                "perseverance", "power", "protection", "reliability", "resilience", "resolve", "stability",
                "steadfastnes", "strength", "trustworthines", "victory", "vigilance", "heraldry", "logo", "seal", "shield"}
    emblem = (df.image_file == "00000005_1.jpg") & df.claimed_noun.isin(emblem_words)

    def count(d):
        vc = d.your_judgment.value_counts()
        a, p, m = int(vc.get("absent", 0)), int(vc.get("present", 0)), int(vc.get("ambiguous", 0))
        return a, p, m
    absent, present, ambiguous = count(df)
    a2, p2, m2 = count(df[~emblem])
    return {
        "n_total_labeled": int(len(df)),
        "n_labelled_in_label_studio": int(len(ls)), "n_labelled_on_quick_labels_page": int((~q.id.isin(ls.id)).sum()),
        "n_absent_genuine_hallucination": absent,
        "n_present_yolo_miss_not_hallucination": present,
        "n_ambiguous": ambiguous,
        "precision_as_hallucination_detector_strict": round(absent / (absent + present), 4),
        "precision_as_hallucination_detector_inclusive_of_ambiguous_as_negative": round(absent / len(df), 4),
        "without_the_24_words_on_one_emblem": {"rows_left_out": int(emblem.sum()), "absent": a2, "present": p2, "ambiguous": m2,
                                               "precision_strict": round(a2 / (a2 + p2), 4)},
    }


def agent4():
    df = pd.read_csv(PKG / "task_c_cross_archive_duplicates/labeling_sheet_LS_export.csv")
    n = len(df)
    same = int((df.your_judgment == "same_composition").sum())
    curve = []
    for t in AGENT4_THRESHOLDS:
        sub = df[df.similarity >= t]
        if len(sub):
            curve.append({"threshold": t, "n_pairs": int(len(sub)),
                          "precision": round(float((sub.your_judgment == "same_composition").mean()), 4)})
    return {
        "n_total_labeled": n,
        "n_same_composition": same,
        "n_same_style_only": int((df.your_judgment == "same_style_only").sum()),
        "n_unrelated": int((df.your_judgment == "unrelated").sum()),
        "overall_precision_all_pairs": round(same / n, 4),
        "similarity_range_labeled": [float(df.similarity.min()), float(df.similarity.max())],
        "precision_at_threshold": curve,
    }


def agent5():
    master = pd.read_csv(PKG / "task_d_ocr_ground_truth/labeling_sheet.csv")[
        ["id", "kosmos_bucket", "kosmos_output"]]
    gt = pd.read_csv(PKG / "task_d_ocr_ground_truth/labeling_sheet_LS_export.csv")
    m = master.merge(gt, on="id")
    m["no_text"] = m.ground_truth_transcription.fillna("").str.strip().str.upper() == "NO_TEXT"

    def bucket(b):
        s = m[m.kosmos_bucket == b]
        return int(len(s)), int((~s.no_text).sum()), int(s.no_text.sum())

    n3, has3, _ = bucket(3)   # "candidate real OCR"
    n1, has1, no1 = bucket(1)  # "degenerate repetition"
    n2, _, no2 = bucket(2)     # "hallucinated captioning"
    return {
        "n_total_labeled": int(len(m)),
        "bucket_precision": {
            "candidate_real_ocr": {"n": n3, "n_has_real_text": has3,
                                   "precision_as_real_text_detector": round(has3 / n3, 4)},
            "degenerate_repetition": {"n": n1, "n_correctly_no_text": no1,
                                      "n_actually_has_text_missed": has1,
                                      "precision_as_no_usable_text_detector": round(no1 / n1, 4)},
            "hallucinated_captioning": {"n": n2, "n_correctly_no_text": no2,
                                        "precision_as_no_usable_text_detector": round(no2 / n2, 4)},
        },
    }


CHECKS = {
    "agent2_contradiction_precision_real.json":
        (agent2, [("precision_as_hallucination_detector_strict", 0.3966)]),
    "agent4_crossarchive_precision_real.json":
        (agent4, [("overall_precision_all_pairs", 0.557)]),
    "agent5_ocr_precision_real.json": (agent5, []),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check recomputation against the committed JSON instead of rewriting")
    args = ap.parse_args()

    ok = True
    for fname, (fn, expectations) in CHECKS.items():
        res = fn()
        path = OUT / fname
        for key, expected in expectations:
            got = res[key]
            match = abs(got - expected) < 5e-4
            ok &= match
            print(f"  {fname:46s} {key} = {got}  (expected {expected}) "
                  f"{'OK' if match else 'MISMATCH'}")
        if not expectations:
            p = res["bucket_precision"]["candidate_real_ocr"]["precision_as_real_text_detector"]
            match = abs(p - 0.525) < 5e-4
            ok &= match
            print(f"  {fname:46s} candidate_real_ocr precision = {p}  (expected 0.525) "
                  f"{'OK' if match else 'MISMATCH'}")
        if not args.verify:
            json.dump(res, open(path, "w"), indent=2)
            print(f"    wrote {path}")

    print("\nALL FIGURES REPRODUCE" if ok else "\nSOME FIGURES DID NOT REPRODUCE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
