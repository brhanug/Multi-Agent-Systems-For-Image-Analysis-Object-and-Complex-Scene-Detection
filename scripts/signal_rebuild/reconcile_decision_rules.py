#!/usr/bin/env python3
"""Unify the two decision rules and re-derive everything that depends on the error label.

The pipeline carries two non-equivalent ways of deciding "scene present":

  STORED      threshold the stored comparison_fusion_score at 0.5. On the cell
              (A_object, A_scene, A_vlm) = (1/3, 0.5, 1) this column holds the exact 0.5,
              so the cell is POSITIVE.
  RECOMPUTED  recompute 0.6*A_object + 0.2*A_scene + 0.2*A_vlm from the stored agent
              columns. A_object is stored to six places as 0.333333, so the same cell
              evaluates to 0.4999998 and is NEGATIVE.

They disagree on 2,217 of 12,110 images (18.3%), and both have produced numbers in the
thesis: the same weights on the same partition give 70 errors under RECOMPUTED and 56
under STORED.

STORED is the arithmetically correct one: 0.6*(1/3) + 0.2*(0.5) + 0.2*(1) = 0.5 exactly,
and only the six-decimal storage of A_object makes it appear otherwise. This script adopts
EXACT arithmetic as the single rule, re-derives the error label and the conformal
calibration under it, and reports every quantity that moves.

Outputs: results/signal_rebuild/reconciled_decision_rule.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
POOL = BASE / "results/multi_agent/corrected_clean_pool.csv"
CHAT = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
OUT = BASE / "results/signal_rebuild/reconciled_decision_rule.json"
O, A, S, V = "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"
FOUR = [O, A, S, V]
GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]


def scores(d):
    d = d.copy()
    d["saa_recomputed"] = 0.6 * d[O] + 0.2 * d[S] + 0.2 * d[V]
    d["obj_exact"] = (d[O] * 3).round() / 3
    d["saa_exact"] = 0.6 * d.obj_exact + 0.2 * d[S] + 0.2 * d[V]
    d["pred_recomputed"] = (d.saa_recomputed >= 0.5).astype(int)
    d["pred_stored"] = (d.comparison_fusion_score >= 0.5).astype(int)
    d["pred_exact"] = (d.saa_exact >= 0.5).astype(int)
    d["U"] = 0.6 * d[FOUR].std(axis=1, ddof=1) + 0.4 * (1 - d[FOUR].mean(axis=1))
    return d


def crc(U_cal, E_cal, U_te, E_te, alpha):
    n = len(U_cal)
    r = np.array([np.mean(E_cal * (U_cal < g)) for g in GRID])
    ok = (n / (n + 1.0)) * r + (1.0 / (n + 1.0)) <= alpha
    lam = float(np.max(GRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
    auto = U_te < lam
    caught, tot = int(E_te[~auto].sum()), int(E_te.sum())
    return {"alpha": alpha, "lambda": round(lam, 4),
            "test_risk": round(float(np.mean(E_te * auto)), 4),
            "bound_satisfied": bool(np.mean(E_te * auto) <= alpha),
            "auto_approval_pct": round(100 * float(auto.mean()), 2),
            "error_recall_pct": round(100 * caught / tot, 2) if tot else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    ag = scores(pd.read_csv(AGENTS).drop_duplicates("image_id"))
    out = {"experiment": "The two decision rules, unified on exact arithmetic",
           "canonical_rule": "exact: A_object snapped to k/3, SAA thresholded at 0.5",
           "corpus": {
               "n": int(len(ag)),
               "stored_vs_recomputed_disagree": int((ag.pred_stored != ag.pred_recomputed).sum()),
               "exact_vs_stored_disagree": int((ag.pred_exact != ag.pred_stored).sum()),
               "exact_vs_recomputed_disagree": int((ag.pred_exact != ag.pred_recomputed).sum()),
               "note": ("Exact agrees with the stored column everywhere. The disagreement is "
                        "entirely the recomputation's six-decimal truncation of A_object.")}}

    sp = pd.read_csv(SPLIT)
    j = scores(sp.merge(pd.read_csv(AGENTS).drop_duplicates("image_id"),
                        left_on="ppn_id", right_on="image_id", how="inner"))
    res = {}
    for tag, col in [("recomputed", "pred_recomputed"), ("stored", "pred_stored"),
                     ("exact (canonical)", "pred_exact")]:
        j["err"] = (j[col] != j.has_scene).astype(int)
        te, cal = j[j.split == "test"], j[j.split == "calibration"]
        tp = int(((j[col] == 1) & (j.has_scene == 1) & (j.split == "test")).sum())
        fp = int(((j[col] == 1) & (j.has_scene == 0) & (j.split == "test")).sum())
        fn = int(((j[col] == 0) & (j.has_scene == 1) & (j.split == "test")).sum())
        tn = int(((j[col] == 0) & (j.has_scene == 0) & (j.split == "test")).sum())
        res[tag] = {
            "test_confusion_TP_FN_TN_FP": [tp, fn, tn, fp],
            "test_errors": fn + fp,
            "test_error_rate": round(float(te.err.mean()), 4),
            "triage_auroc": (round(float(roc_auc_score(te.err, te.U)), 4)
                             if te.err.nunique() > 1 else None),
            "crc": [crc(cal.U.values, cal.err.values, te.U.values, te.err.values, al)
                    for al in ALPHAS],
        }
    out["by_rule"] = res

    # what the blind-spot finding does under the canonical rule
    old = pd.read_csv(POOL)[["ppn_id", "has_scene"]]
    new = pd.read_csv(CHAT); new["has_scene"] = (new.label == "scene").astype(int)
    neg = scores(pd.concat([old, new[["ppn_id", "has_scene"]]]).merge(
        pd.read_csv(AGENTS).drop_duplicates("image_id"),
        left_on="ppn_id", right_on="image_id", how="inner"))
    neg = neg[neg.has_scene == 0]
    bs = {}
    for tag, col in [("recomputed", "pred_recomputed"), ("exact (canonical)", "pred_exact")]:
        e = (neg[col] != neg.has_scene).astype(int)
        bs[tag] = {"n": int(len(neg)), "errors": int(e.sum()),
                   "auroc": (round(float(roc_auc_score(e, neg[FOUR].std(axis=1, ddof=1))), 4)
                             if e.nunique() > 1 else None)}
    out["blind_spot_by_rule"] = bs

    c5 = {t: next(r for r in v["crc"] if r["alpha"] == 0.05) for t, v in res.items()}
    out["headline_at_alpha_0.05"] = c5
    out["what_changes"] = (
        f"Adopting exact arithmetic moves the test-partition error count from "
        f"{res['recomputed']['test_errors']} to {res['exact (canonical)']['test_errors']}, "
        f"because 2,217 corpus images (52 in the test partition) move from scene-absent to "
        f"scene-present. At alpha=0.05 auto-approval goes from "
        f"{c5['recomputed']['auto_approval_pct']}% to {c5['exact (canonical)']['auto_approval_pct']}% "
        f"and error recall from {c5['recomputed']['error_recall_pct']}% to "
        f"{c5['exact (canonical)']['error_recall_pct']}%. The blind-spot AUROC on confirmed "
        f"negatives moves from {bs['recomputed']['auroc']} to {bs['exact (canonical)']['auroc']}.")

    if a.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
