#!/usr/bin/env python3
"""Can the fixed fusion weights (0.6, 0.2, 0.2) be replaced by learned ones?

Section 4.2.1 compares five hand-picked weightings. None was actually fitted, so the
question of what a *learned* weighting would do was left open. This settles it, and
does so exhaustively rather than by search.

The key structural fact: A_object takes 4 values, A_scene takes 2, and
A_vlm = 1[A_object > 0] exactly, so the three fusion inputs realise only 8 distinct
triples across the whole 1,250-image pool. Any score of the form
    SAA = w_o*A_object + w_s*A_scene + w_v*A_vlm,  thresholded at 0.5,
therefore induces one of at most 2^8 labellings of those 8 cells -- and only those that
are linearly realisable by some admissible w. Each candidate labelling is checked for
realisability by linear programming, so the result is the true optimum over ALL
weightings, not the best found by a grid.

Two weight families are reported:
  simplex  - w >= 0, sum(w) = 1, threshold 0.5 (the deployed parameterisation)
  freeform - w >= 0 unconstrained in scale, learned threshold (a general linear rule)

Protocol: pick the best realisable labelling on the train partition, report its test
performance. The best achievable on test is also reported as an oracle ceiling, to show
how much of the gap is estimation error rather than representational limit.

Outputs: results/signal_rebuild/learned_fusion_weights.json
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE = Path("/data/brhanu/thesis_project")
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/learned_fusion_weights.json"
DEPLOYED = (0.6, 0.2, 0.2)
EPS = 1e-6


def cells(df):
    """The distinct (A_object, A_scene, A_vlm) triples and their class counts."""
    g = (df.groupby(["existing_pipeline_agent", "scene_agent", "vlm_agent"])
           .has_scene.agg(pos="sum", n="size").reset_index())
    g["neg"] = g.n - g.pos
    return g


def realisable(x, want, family):
    """Is there an admissible w with w.x >= 0.5 exactly on the cells `want` marks 1?

    simplex : w >= 0, sum w = 1, threshold fixed at 0.5.
    freeform: w >= 0, learned threshold t; scale fixed by sum(w)=1 without loss of
              generality since the rule is scale-invariant once t is free.
    """
    m = len(x)
    if family == "simplex":
        A_ub, b_ub = [], []
        for i in range(m):
            if want[i]:                       # w.x >= 0.5  ->  -w.x <= -0.5
                A_ub.append(-x[i]); b_ub.append(-0.5)
            else:                             # w.x <= 0.5 - eps
                A_ub.append(x[i]); b_ub.append(0.5 - EPS)
        r = linprog(c=np.zeros(3), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                    A_eq=np.ones((1, 3)), b_eq=[1.0],
                    bounds=[(0, None)] * 3, method="highs")
        return bool(r.success), (list(np.round(r.x, 4)) if r.success else None), 0.5
    # freeform: variables (w_o, w_s, w_v, t)
    A_ub, b_ub = [], []
    for i in range(m):
        row = list(x[i]) + [-1.0]
        if want[i]:                           # w.x - t >= 0   ->  -(w.x - t) <= 0
            A_ub.append([-v for v in row]); b_ub.append(0.0)
        else:                                 # w.x - t <= -eps
            A_ub.append(row); b_ub.append(-EPS)
    r = linprog(c=np.zeros(4), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                A_eq=np.array([[1, 1, 1, 0]]), b_eq=[1.0],
                bounds=[(0, None)] * 3 + [(None, None)], method="highs")
    return bool(r.success), (list(np.round(r.x[:3], 4)) if r.success else None), \
           (round(float(r.x[3]), 4) if r.success else None)


def score(want, cel):
    """Confusion counts for a labelling of the cells."""
    tp = int(sum(cel.pos[i] for i in range(len(cel)) if want[i]))
    fp = int(sum(cel.neg[i] for i in range(len(cel)) if want[i]))
    fn = int(cel.pos.sum() - tp); tn = int(cel.neg.sum() - fp)
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "errors": fn + fp,
            "balanced_acc": round((rec + spec) / 2, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "degenerate": bool(tp + fp == 0 or fn + tn == 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    sp = pd.read_csv(SPLIT)
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    tr, te = j[j.split == "train"], j[j.split == "test"]
    ctr, cte = cells(tr), cells(te)
    x = ctr[["existing_pipeline_agent", "scene_agent", "vlm_agent"]].values
    assert (cte[["existing_pipeline_agent", "scene_agent", "vlm_agent"]].values == x).all()

    def deployed(c):
        s = c[["existing_pipeline_agent", "scene_agent", "vlm_agent"]].values @ np.array(DEPLOYED)
        return score((s >= 0.5).astype(int), c)

    out = {"experiment": "Exhaustive optimum over all fusion weightings",
           "distinct_signal_triples": int(len(x)),
           "why_exhaustive": ("A_object has 4 values, A_scene 2, and A_vlm = 1[A_object>0], so the "
                              "three fusion inputs realise only 8 triples. Every weighting induces "
                              "one of at most 2^8 cell labellings; each is checked for linear "
                              "realisability by LP, so this is the true optimum, not a grid search."),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "deployed_weights": list(DEPLOYED),
           "deployed_train": deployed(ctr), "deployed_test": deployed(cte)}

    for family in ("simplex", "freeform"):
        cands = []
        for want in itertools.product([0, 1], repeat=len(x)):
            ok, w, t = realisable(x, want, family)
            if not ok:
                continue
            cands.append({"cells": list(want), "weights": w, "threshold": t,
                          "train": score(want, ctr), "test": score(want, cte)})
        best_tr = min(cands, key=lambda c: c["train"]["errors"])
        oracle = min(cands, key=lambda c: c["test"]["errors"])
        fp_budget = out["deployed_train"]["FP"]
        cons = [c for c in cands if c["train"]["FP"] <= fp_budget]
        best_cons = min(cons, key=lambda c: c["train"]["errors"]) if cons else None
        nondeg = [c for c in cands if not c["train"]["degenerate"]]
        best_bal = max(cands, key=lambda c: c["train"]["balanced_acc"])
        best_bal_nd = max(nondeg, key=lambda c: c["train"]["balanced_acc"]) if nondeg else None
        out[family] = {
            "realisable_classifiers": len(cands),
            "non_degenerate_classifiers": len(nondeg),
            "best_by_train_errors": best_tr,
            "best_by_train_balanced_acc": best_bal,
            "best_non_degenerate_by_balanced_acc": best_bal_nd,
            "oracle_best_on_test": {"test": oracle["test"], "weights": oracle["weights"]},
            "best_at_deployed_FP_budget": best_cons,
            "fp_budget_used": fp_budget,
            "all_realisable_summary": [
                {"cells": c["cells"], "weights": c["weights"], "threshold": c["threshold"],
                 "train_errors": c["train"]["errors"],
                 "train_bal_acc": c["train"]["balanced_acc"],
                 "test_errors": c["test"]["errors"],
                 "test_bal_acc": c["test"]["balanced_acc"],
                 "test_FP": c["test"]["FP"], "test_FN": c["test"]["FN"],
                 "degenerate": c["train"]["degenerate"]}
                for c in sorted(cands, key=lambda c: -c["train"]["balanced_acc"])],
        }

    # ---- the knife-edge cell: a truncation artifact deciding a fifth of the corpus ----
    jj = j.copy()
    jj["saa_stored"] = jj[["existing_pipeline_agent", "scene_agent", "vlm_agent"]].values @ np.array(DEPLOYED)
    jj["obj_exact"] = (jj.existing_pipeline_agent * 3).round() / 3
    jj["saa_exact"] = 0.6 * jj.obj_exact + 0.2 * jj.scene_agent + 0.2 * jj.vlm_agent
    edge = ((jj.obj_exact - 1 / 3).abs() < 1e-9) & (jj.scene_agent == 0.5)

    def conf(d, col):
        pred = (d[col] >= 0.5).values; hs = (d.has_scene == 1).values
        tp = int((pred & hs).sum()); fp = int((pred & ~hs).sum())
        fn = int((~pred & hs).sum()); tn = int((~pred & ~hs).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "errors": fn + fp,
                "balanced_acc": round((rec + spec) / 2, 4)}

    alts = {"deployed_0.6_0.2_0.2": DEPLOYED, "uniform": (1 / 3, 1 / 3, 1 / 3),
            "0.5_0.25_0.25": (0.5, 0.25, 0.25), "0.4_0.3_0.3": (0.4, 0.3, 0.3),
            "shapley_0.28_0.19_0.33": (0.28, 0.19, 0.33)}
    cs = np.array([0.333333, 0.5, 1.0]); ce = np.array([1 / 3, 0.5, 1.0])
    out["knife_edge_cell"] = {
        "cell": "A_object=1/3, A_scene=0.5, A_vlm=1",
        "n_full_pool": int(edge.sum()),
        "pct_full_pool": round(100 * float(edge.mean()), 1),
        "n_test": int((edge & (jj.split == "test")).sum()),
        "saa_as_stored": 0.4999998, "saa_with_exact_third": 0.5,
        "stored_A_object_value": 0.333333,
        "deployed_prediction_as_stored": "negative (misses threshold by 2e-7)",
        "deployed_prediction_if_exact": "positive",
        "per_weighting_on_this_cell": {
            k: {"saa_stored": round(float(np.array(w) @ cs), 7),
                "positive_stored": bool(np.array(w) @ cs >= 0.5),
                "saa_exact": round(float(np.array(w) @ ce), 7),
                "positive_exact": bool(np.array(w) @ ce >= 0.5)}
            for k, w in alts.items()},
        "test_confusion_as_stored": conf(jj[jj.split == "test"], "saa_stored"),
        "test_confusion_if_exact": conf(jj[jj.split == "test"], "saa_exact"),
        "finding": ("The deployed weighting is the only one of the five in Table 4.2 that leaves "
                    "this cell below threshold, and it misses by 2e-7 purely because A_object is "
                    "stored as 0.333333 rather than 1/3. Recomputed with exact thirds the deployed "
                    "weighting reproduces 179/32/14/24 and 56 errors -- the identical figures the "
                    "table reports for all four alternatives. The apparent conservative/aggressive "
                    "trade-off between deployed and alternative weightings is therefore this one "
                    "truncation, not a property of the weights."),
    }

    # does any admissible weighting rescue the A_object=0 scene-positive images?
    z = j[(j.existing_pipeline_agent == 0) & (j.has_scene == 1)]
    out["structural_limit"] = {
        "A_object_0_and_scene_present": int(len(z)),
        "note": ("These need w_s*A_scene >= threshold on their own. Reachable only when the "
                 "scene weight is large enough, which is exactly the trade the optimum makes "
                 "or declines to make -- see whether the best weights put w_s at or above 0.5.")}

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
