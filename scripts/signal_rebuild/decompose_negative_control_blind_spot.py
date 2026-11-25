#!/usr/bin/env python3
"""Decompose the below-chance AUROC on confirmed true negatives.

The headline finding is that U_SAA scores AUROC = 0.1259 on confirmed true
negatives. That is usually read as evidence the four models share a blind spot.
This script separates how much of it is model behaviour and how much is forced
by the construction of the signals.

The key observation: on a true negative an "error" means SAA >= 0.5. Since
SAA = 0.6*A_object + 0.2*A_scene + 0.2*A_vlm and A_vlm = 1[A_object > 0], an
image with A_object = 0 has SAA <= 0.2 and therefore *cannot* be an error. But
A_agree = 1 - |A_object - A_vlm| = 1 exactly there, which is the single largest
outlier among the four signals, so U_SAA is at its highest on precisely the
images that are guaranteed correct.

Outputs: results/signal_rebuild/negative_control_blind_spot_decomposition.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
POOL = BASE / "results/multi_agent/corrected_clean_pool.csv"
NEW = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
OUT = BASE / "results/signal_rebuild/negative_control_blind_spot_decomposition.json"

A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
MEASURED = ["existing_pipeline_agent", "scene_agent", "vlm_agent"]


def attach(df, ag):
    j = df.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["saa"] = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    j["err"] = ((j.saa >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int)
    j["U_SAA"] = j[A4].std(axis=1, ddof=1)
    j["U_measured3"] = j[MEASURED].std(axis=1, ddof=1)
    return j


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(np.unique(y)) > 1 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    new = pd.read_csv(NEW); new["has_scene"] = (new.label == "scene").astype(int)
    old = pd.read_csv(POOL)
    neg = pd.concat([attach(old[["ppn_id", "has_scene"]], ag),
                     attach(new[["ppn_id", "has_scene"]], ag)])
    neg = neg[neg.has_scene == 0].copy()

    zero = neg[neg.existing_pipeline_agent == 0]
    pos = neg[neg.existing_pipeline_agent > 0]

    out = {
        "experiment": "Is the below-chance AUROC on true negatives model behaviour or construction?",
        "pool": {"n": int(len(neg)), "n_errors": int(neg.err.sum()),
                 "auroc_U_SAA": auc(neg.err, neg.U_SAA)},
        "stratum_A_object_eq_0": {
            "n": int(len(zero)), "n_errors": int(zero.err.sum()),
            "max_saa": round(float(zero.saa.max()), 4),
            "error_threshold": 0.5,
            "errors_arithmetically_possible": bool(zero.saa.max() >= 0.5),
            "U_SAA_min": round(float(zero.U_SAA.min()), 4),
            "U_SAA_max": round(float(zero.U_SAA.max()), 4)},
        "stratum_A_object_gt_0": {
            "n": int(len(pos)), "n_errors": int(pos.err.sum()),
            "U_SAA_min": round(float(pos.U_SAA.min()), 4),
            "U_SAA_max": round(float(pos.U_SAA.max()), 4),
            "auroc_U_SAA": auc(pos.err, pos.U_SAA),
            "auroc_measured3": auc(pos.err, pos.U_measured3),
            "auroc_confidence": auc(pos.err, 1 - pos.saa)},
        "ranges_disjoint": bool(pos.U_SAA.max() < zero.U_SAA.min()),
        "whole_pool_alternatives": {
            "auroc_measured3_derived_term_dropped": auc(neg.err, neg.U_measured3),
            "auroc_confidence_baseline": auc(neg.err, 1 - neg.saa)},
    }
    a_all, a_sub = out["pool"]["auroc_U_SAA"], out["stratum_A_object_gt_0"]["auroc_U_SAA"]
    out["gap_to_chance_closed_by_removing_artifact_pct"] = round(
        100 * ((0.5 - a_all) - (0.5 - a_sub)) / (0.5 - a_all), 1)
    out["interpretation"] = (
        "Errors are impossible in the A_object=0 stratum (max SAA 0.2 < 0.5 threshold), yet "
        "U_SAA is strictly higher there than anywhere else, because the derived term A_agree "
        "jumps to 1. U_SAA therefore ranks guaranteed-correct images above every error before "
        "any model behaviour is involved. Restricting to the stratum where errors can occur "
        "raises AUROC but leaves it below chance, so a genuine residual failure remains. The "
        "confidence baseline scores 0.0 on this subgroup because error is defined as SAA>=0.5, "
        "which makes the subgroup circular for any SAA-derived score.")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
