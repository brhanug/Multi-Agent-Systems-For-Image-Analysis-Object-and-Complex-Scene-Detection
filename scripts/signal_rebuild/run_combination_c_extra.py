#!/usr/bin/env python3
"""Extra analysis, added after S2 of PLAN_selection.md showed the joint mapping is sensitive to its
regularisation: are the fixed-weight combination and the describer alone sensitive in the same way?
Per-agent mappings at C = 0.1, 1 and 10; same 992 pictures, labels and 200 redraws.

    python run_combination_c_extra.py [--verify]
"""
import argparse, json

import numpy as np
from sklearn.linear_model import LogisticRegression

import run_best_config_repeated as rb
import run_scene_classes as rs
from run_more_experiments import redraws

OUT = rb.SR / "combination_c_extra.json"
K = rs.K


def probs(X, rf, yf, rall, C):
    m = LogisticRegression(max_iter=5000, C=C).fit(X[rf], yf)
    P = np.zeros((len(rall), K)); P[:, m.classes_] = m.predict_proba(X[rall]); return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    res = {f"{v} C={C}": [] for v in ("fixed 0.6/0.2/0.2", "describer alone") for C in (0.1, 1.0, 10.0)}
    for tr, cal, te in redraws(d.ppn.values):
        for C in (0.1, 1.0, 10.0):
            Pk = {k: probs(V[k], rows[tr], y[tr], rows[te], C) for k in V}
            for v, P in (("fixed 0.6/0.2/0.2", 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"]), ("describer alone", Pk["vlm"])):
                pt, yt = P.argmax(1), y[te]
                res[f"{v} C={C}"].append(float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any()])))
    out = {"extra": "added after S2 of PLAN_selection.md (commit 42db659)", "balanced_accuracy": {k: rb.summarise(v) for k, v in res.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print({k: v["mean"] for k, v in out["balanced_accuracy"].items()}); return 0


if __name__ == "__main__":
    raise SystemExit(main())
