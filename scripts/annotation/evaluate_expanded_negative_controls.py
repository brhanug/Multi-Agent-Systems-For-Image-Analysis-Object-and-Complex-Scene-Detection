"""
Re-evaluate the shared-blind-spot finding on the expanded negative-control pool.

The thesis reports that on confirmed true negatives the disagreement signal ranks
errors below chance (ROC-AUC 0.1152), on 38 images carrying 5 errors, with a wide
interval. This recomputes that on the images newly reviewed in the contact-sheet
pass, and on everything confirmed to date.

    python evaluate_expanded_negative_controls.py [--verify]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, csv, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
NEW = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
POOL = BASE / "results/multi_agent/corrected_clean_pool.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/multi_agent/expanded_negative_control_evaluation.json"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
SEED, N_BOOT = 0, 4000


def boot_auc(y, s, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], s[i]))
    o = np.array(out)
    return {"auc": round(float(roc_auc_score(y, s)), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)],
            "entirely_below_chance": bool(np.percentile(o, 97.5) < 0.5),
            "n": int(len(y)), "n_errors": int(y.sum()), "n_resamples": int(len(o))}


def attach(df, ag):
    j = df.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["saa_fusion"] = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    j["err"] = ((j.saa_fusion >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int)
    j["U_SAA"] = j[A4].std(axis=1, ddof=1)
    return j


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")

    new = pd.read_csv(NEW)
    new["has_scene"] = (new.label == "scene").astype(int)
    new_j = attach(new[["ppn_id", "has_scene"]], ag)

    old = pd.read_csv(POOL)
    old_j = attach(old[["ppn_id", "has_scene"]], ag)

    out = {"experiment": "Shared blind spot on an expanded confirmed-negative pool",
           "new_labels_this_pass": int(len(new)),
           "new_confirmed_negatives": int((new.has_scene == 0).sum()),
           "new_confirmed_positives": int((new.has_scene == 1).sum()),
           "catalogue_default_wrong_pct": round(100 * float((new.has_scene == 1).mean()), 1)}

    blocks = {
        "existing_pool_all_confirmed_negatives": old_j[old_j.has_scene == 0],
        "new_pass_confirmed_negatives": new_j[new_j.has_scene == 0],
        "combined_confirmed_negatives": pd.concat([old_j[old_j.has_scene == 0],
                                                   new_j[new_j.has_scene == 0]]),
    }
    out["negative_control_blind_spot"] = {}
    for name, d in blocks.items():
        y = d.err.values
        if len(np.unique(y)) < 2:
            out["negative_control_blind_spot"][name] = {
                "n": int(len(d)), "n_errors": int(y.sum()),
                "note": "single class, AUROC undefined"}
            continue
        r = boot_auc(y, d.U_SAA.values)
        r["false_positive_rate"] = round(float((d.saa_fusion >= SAA_THRESHOLD).mean()), 4)
        out["negative_control_blind_spot"][name] = r

    # does the pipeline's false-positive behaviour on negatives hold up?
    out["pipeline_on_negatives"] = {
        k: {"n": int(len(d)), "false_positives": int((d.saa_fusion >= SAA_THRESHOLD).sum()),
            "fp_rate": round(float((d.saa_fusion >= SAA_THRESHOLD).mean()), 4)}
        for k, d in blocks.items()}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
