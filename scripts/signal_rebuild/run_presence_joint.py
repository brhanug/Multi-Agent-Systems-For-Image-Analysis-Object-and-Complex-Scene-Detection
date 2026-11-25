#!/usr/bin/env python3
"""self_annotation_package/more_experiments/PLAN_presence_joint.md: one joint learned mapping on all 22
agent numbers for scene presence, against the fixed 0.6/0.2/0.2, on the same 200 redraws.

    python run_presence_joint.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb

OUT = rb.SR / "presence_joint.json"
ALPHAS = [0.05, 0.10]
KEYS = ("object", "scene", "vlm")


def nb(xs, ratio=0.2 / 0.6):
    xs = np.asarray(xs); m = xs.mean(); se = np.sqrt((1 / len(xs) + ratio) * xs.var(ddof=1)); t = stats.t.ppf(0.975, len(xs) - 1)
    return [round(float(m - t * se), 4), round(float(m + t * se), 4)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    X = np.column_stack([V[k] for k in KEYS])
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    acc = {r: {"scene_auroc": [], "test_error": [], **{f"approved@{al}": [] for al in ALPHAS}, **{f"risk@{al}": [] for al in ALPHAS}}
           for r in ("fixed", "joint")}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in KEYS])
        joint = LogisticRegression(max_iter=2000, C=1.0).fit(X[rows[tr]], y[tr]).predict_proba(X[rows])[:, 1]
        for r, s in (("fixed", P @ rb.W), ("joint", joint)):
            err = ((s >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(s - 0.5)
            acc[r]["scene_auroc"].append(float(roc_auc_score(y[te], s[te]))); acc[r]["test_error"].append(float(err[te].mean()))
            for al in ALPHAS:
                lam = rb.lam_for(u[cal], err[cal], al); auto = u[te] < lam
                acc[r][f"approved@{al}"].append(float(auto.mean())); acc[r][f"risk@{al}"].append(float(np.mean(err[te] * auto)))
    out = {"plan": "self_annotation_package/more_experiments/PLAN_presence_joint.md", "pictures": int(len(sp)), "draws": rb.N_DRAWS,
           "per_rule": {r: {m: rb.summarise(v) for m, v in z.items()} for r, z in acc.items()},
           "joint_minus_fixed": {}}
    for m in ("scene_auroc", "test_error", "approved@0.05", "approved@0.1"):
        d = np.array(acc["joint"][m]) - np.array(acc["fixed"][m])
        out["joint_minus_fixed"][m] = {**rb.paired(d), "corrected_t_95": nb(d)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for r, z in out["per_rule"].items(): print(r, {m: v["mean"] for m, v in z.items()})
    for m, v in out["joint_minus_fixed"].items(): print(" joint-fixed", m, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
