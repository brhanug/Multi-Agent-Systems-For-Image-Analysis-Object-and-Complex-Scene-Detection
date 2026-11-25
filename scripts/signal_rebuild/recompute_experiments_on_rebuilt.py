#!/usr/bin/env python3
"""Recompute the experiments that consume the four agent signals, on rebuilt signals.

The full Tier 1 re-run replaced the deployed signal set -- in which A_object came from a
manifest fallback and A_vlm was 2/4 on 98.9% of images -- with signals measured by real
models on all 12,110 images. Several published experiments still ran on the old set. This
recomputes the ones that actually consume those signals and reports both side by side.

Scope, stated precisely rather than claimed broadly:
  AWLF v2            recomputed, with a caveat -- the deployed feature vector includes a
                     restoration_agent column that the re-run does not reproduce, so the
                     rebuilt version uses five of its six features.
  Tier 1 ablation    recomputed in full (Object, Scene, VLM, OCR all rebuilt), both as
                     mean spread (the deployed measure) and predictively.
  Shapley            recomputed in full (all four coalition members rebuilt).
  CRC                already recomputed, with its own recalibration, in
                     run_end_to_end_rebuilt.py -- not repeated here.
  Spatial meta-learner (Exp 31)  operates on box-level IoU, not on these scalars; the
                     rebuild does not change its inputs and it is not recomputed.

Outputs: results/signal_rebuild/recomputed_on_rebuilt.json
"""
import argparse, itertools, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = SR / "recomputed_on_rebuilt.json"
SEED = 20260912
DEP = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
REB = ["A_object", "A_agree", "A_scene", "A_vlm"]


def load():
    reb = pd.read_csv(SR / "full_tier1_signals.csv")
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    sp = pd.read_csv(SPLIT)
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner").merge(
        reb, on="image_id", how="inner")
    j["saa_dep"] = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    j["err_dep"] = ((j.saa_dep >= 0.5).astype(int) != j.has_scene).astype(int)
    j["saa_reb"] = 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm
    cal = j[j.split == "calibration"]
    grid = np.arange(0.20, 0.95, 0.005)
    thr = float(grid[int(np.argmin([float(((cal.saa_reb >= t).astype(int) != cal.has_scene).mean())
                                    for t in grid]))])
    j["err_reb"] = ((j.saa_reb >= thr).astype(int) != j.has_scene).astype(int)
    j["ppn"] = j.ppn_id.str.split("/").str[0]
    return j, thr


def cv_auc(d, cols, target):
    X = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = d[target].values.astype(int)
    g = d.ppn.values
    if len(np.unique(y)) < 2:
        return None
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=SEED))
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return round(float(roc_auc_score(y[ok], oof[ok])), 4) if len(np.unique(y[ok])) > 1 else None


def shapley(d, cols, target):
    """Exact Shapley values over the coalition of signals, value = CV ROC-AUC."""
    n = len(cols)
    val = {}
    for r in range(n + 1):
        for c in itertools.combinations(range(n), r):
            val[c] = 0.5 if not c else (cv_auc(d, [cols[i] for i in c], target) or 0.5)
    from math import factorial
    phi = {}
    for i in range(n):
        tot = 0.0
        for r in range(n):
            for c in itertools.combinations([k for k in range(n) if k != i], r):
                w = factorial(len(c)) * factorial(n - len(c) - 1) / factorial(n)
                tot += w * (val[tuple(sorted(c + (i,)))] - val[c])
        phi[cols[i]] = round(tot, 4)
    return phi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    j, thr = load()
    out = {"experiment": "Experiments recomputed on the rebuilt Tier 1 signals",
           "n": int(len(j)), "rebuilt_decision_threshold": round(thr, 3),
           "scope_note": ("CRC is recomputed with its own recalibration in "
                          "run_end_to_end_rebuilt.py. The Exp 31 spatial meta-learner runs on "
                          "box-level IoU rather than these scalars and is unaffected.")}

    # ---- AWLF v2 ----
    dep_feats = [c for c in DEP + ["document_agent", "restoration_agent"] if c in j.columns]
    reb_feats = [c for c in REB + ["A_ocr"] if c in j.columns]
    out["awlf_v2"] = {
        "deployed_features": dep_feats, "deployed_auroc": cv_auc(j, dep_feats, "err_dep"),
        "rebuilt_features": reb_feats, "rebuilt_auroc": cv_auc(j, reb_feats, "err_reb"),
        "caveat": ("The deployed vector has a restoration_agent feature the re-run does not "
                   "reproduce, so the rebuilt model uses five features against six. The "
                   "comparison is indicative, not like-for-like.")}

    # ---- Tier 1 ablation, both ways ----
    dep4 = ["existing_pipeline_agent", "scene_agent", "vlm_agent", "document_agent"]
    reb4 = ["A_object", "A_scene", "A_vlm", "A_ocr"]
    abl = {}
    for tag, cols, target in [("deployed", dep4, "err_dep"), ("rebuilt", reb4, "err_reb")]:
        cols = [c for c in cols if c in j.columns]
        full_sp = j[cols].std(axis=1, ddof=1)
        rows = {"full": {"mean_spread": round(float(full_sp.mean()), 4),
                         "predictive_auroc": cv_auc(j, cols, target)}}
        for drop in cols:
            keep = [c for c in cols if c != drop]
            rows[f"drop {drop}"] = {
                "mean_spread": round(float(j[keep].std(axis=1, ddof=1).mean()), 4),
                "predictive_auroc": cv_auc(j, keep, target)}
        abl[tag] = rows
    out["tier1_ablation"] = abl

    # ---- Shapley ----
    out["shapley"] = {
        "deployed": shapley(j, DEP, "err_dep"),
        "rebuilt": shapley(j, REB, "err_reb"),
        "value_function": "5-fold grouped CV ROC-AUC of a logistic model on the coalition"}

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
