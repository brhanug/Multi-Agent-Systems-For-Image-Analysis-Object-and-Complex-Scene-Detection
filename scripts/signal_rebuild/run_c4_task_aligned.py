#!/usr/bin/env python3
"""C4 follow-up, added after the registered test (23 Nov 2026, prompted by an outside review).

The registered test (run_c4_preregistered.py) ranks errors of the fused SCENE call with a disagreement
signal built from the OBJECT-class scores, so signal and error belong to different tasks. This
follow-up measures disagreement against confidence within each task, on the same 300 hand-annotated
pictures, the same score matrix and the same cross-fitted Platt calibration (same folds and seed):

  objects  one decision per (picture, class): fused probability = mean of the three systems' calibrated
           per-class probabilities, call at 0.5, error = call differs from the human class label.
           disagreement = spread (population std) of the three calibrated probabilities for that class;
           confidence   = 1 - |fused probability - 0.5|.
           AUROC over all 2,400 decisions; the 95% interval of the difference comes from a bootstrap
           over pictures (the eight decisions of one picture are not independent).
  scenes   error = the fused scene call (0.6/0.2/0.2 at 0.5) differs from the human label, as in C4.
           disagreement = spread of the three scores that make up that call (object finder, scene
           matcher, describer): raw, and after cross-fitted Platt calibration against the scene label;
           confidence = 1 - |fused score - 0.5| (C4's signal 5). DeLong 95% interval.

It changes nothing in the registered result; it says whether that result depends on the task mismatch.

    python run_c4_task_aligned.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import run_c4_preregistered as c4

BASE = c4.BASE
OUT = BASE / "results/signal_rebuild/c4/c4_task_aligned.json"
SCENE_PARTS = ["existing_pipeline_agent", "scene_agent", "vlm_agent"]
N_BOOT, BOOT_SEED = 2000, 20261123


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(c4.CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    R = m["raw"][:, [ms.index(s) for s in c4.SYSTEMS]][:, :, [mc.index(c) for c in c4.CLASSES]].astype(np.float64)
    truth = pd.read_csv(c4.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    ag = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv").drop_duplicates("image_id").set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); n = len(rows); tt = truth.reset_index(drop=True)
    folds = list(KFold(5, shuffle=True, random_state=c4.SEED).split(np.arange(n)))   # the folds of C4

    # objects: per (picture, class)
    P = np.zeros((n, len(c4.SYSTEMS), len(c4.CLASSES)))
    for tr, te in folds:
        P[te] = c4.apply_calibrators(c4.fit_calibrators(R[rows[tr]], tt.iloc[tr]), R[rows[te]])
    fused = P.mean(axis=1); Y = tt[c4.CLASSES].values
    err = ((fused >= 0.5).astype(int) != Y).astype(int)
    dis = P.std(axis=1); conf = 1 - np.abs(fused - 0.5)
    e, d, c = err.ravel(), dis.ravel(), conf.ravel()
    au_d, au_c = roc_auc_score(e, d), roc_auc_score(e, c)
    rng = np.random.default_rng(BOOT_SEED); diffs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n); eb = err[s].ravel()
        if 0 < eb.sum() < len(eb):
            diffs.append(roc_auc_score(eb, dis[s].ravel()) - roc_auc_score(eb, conf[s].ravel()))
    per_class = {}
    for ci, cl in enumerate(c4.CLASSES):
        ec = err[:, ci]
        per_class[cl] = ({"errors": int(ec.sum()), "disagreement": round(float(roc_auc_score(ec, dis[:, ci])), 4),
                          "confidence": round(float(roc_auc_score(ec, conf[:, ci])), 4)} if 0 < ec.sum() < n else {"errors": int(ec.sum())})
    objects = {"decisions": int(e.size), "errors": int(e.sum()),
               "auroc": {"disagreement": round(float(au_d), 4), "confidence": round(float(au_c), 4)},
               "difference": round(float(au_d - au_c), 4),
               "bootstrap_ci95_over_pictures": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
               "classes_where_disagreement_higher": [k for k, v in per_class.items() if "disagreement" in v and v["disagreement"] > v["confidence"]],
               "per_class": per_class}

    # scenes: the three parts of the fused scene call
    S = ag.loc[truth.index, SCENE_PARTS].values.astype(np.float64)
    fs = 0.6 * S[:, 0] + 0.2 * S[:, 1] + 0.2 * S[:, 2]; ys = truth["has_scene"].values
    es = ((fs >= 0.5).astype(int) != ys).astype(int)
    Sc = np.zeros_like(S)
    for tr, te in folds:
        for j in range(S.shape[1]):
            lr = LogisticRegression(C=100.0).fit(c4.logit(S[tr, j]).reshape(-1, 1), ys[tr])
            Sc[te, j] = lr.predict_proba(c4.logit(S[te, j]).reshape(-1, 1))[:, 1]
    conf_s = 1 - np.abs(fs - 0.5)
    scenes = {"pictures": int(n), "errors": int(es.sum()), "auroc": {}, "difference_vs_confidence": {}}
    for name, sig in (("disagreement_raw", S.std(axis=1)), ("disagreement_calibrated", Sc.std(axis=1)), ("confidence", conf_s)):
        scenes["auroc"][name] = round(float(roc_auc_score(es, sig)), 4)
        if name != "confidence":
            dd, ci, p = c4.delong(es, sig, conf_s)
            scenes["difference_vs_confidence"][name] = {"difference": round(dd, 4), "delong_ci95": [round(ci[0], 4), round(ci[1], 4)]}

    out = {"experiment": "C4 follow-up: disagreement against confidence within each task (added after the registered test)",
           "objects_per_picture_and_class": objects, "scene_presence": scenes,
           "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED, "kept": len(diffs)}}
    out = json.loads(json.dumps(out))
    if a.verify:
        same = json.loads(OUT.read_text()) == out; print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
