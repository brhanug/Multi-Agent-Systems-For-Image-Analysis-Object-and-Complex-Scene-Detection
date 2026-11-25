#!/usr/bin/env python3
"""Is the corrected design's probability of a scene calibrated?

The corrected design calls a picture a scene when its combined probability p reaches 0.5 and routes it
by |p - 0.5|, so p is meant as a probability. Each agent's mapping is a logistic fit, which is
calibrated on its training data, but the fixed 0.6/0.2/0.2 average of three calibrated probabilities
need not be. Over the same 200 book-disjoint redraws as best_config_repeated.json, on the test books:

  ECE    expected calibration error, 10 equal-width bins (and 15 for comparison)
  Brier  mean squared error of p against the human label, beside the base rate's
  for    the combined p, each agent's own probability, and p recalibrated by Platt scaling on the
         calibration books (to see how much a recalibration step would buy)

plus a pooled reliability table: in each bin of p, the mean p and the observed share of scenes.

    python run_corrected_calibration.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_best_config_repeated as rb

OUT = rb.SR / "corrected_calibration.json"
BINS = 10


def ece(p, y, bins=BINS):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    V = {**nat, "scene": scene_abs}
    names = ["combined", "object", "scene", "describer", "combined, Platt-recalibrated"]
    acc = {n: {"ece": [], "ece15": [], "brier": []} for n in names}; base = []
    pooled_p, pooled_y = [], []
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
        p = P @ rb.W
        pl = LogisticRegression(C=100.0).fit(logit(p[cal]).reshape(-1, 1), y[cal]).predict_proba(logit(p).reshape(-1, 1))[:, 1]
        for n, q in zip(names, [p, P[:, 0], P[:, 1], P[:, 2], pl]):
            acc[n]["ece"].append(ece(q[te], y[te])); acc[n]["ece15"].append(ece(q[te], y[te], 15))
            acc[n]["brier"].append(float(((q[te] - y[te]) ** 2).mean()))
        base.append(float(((y[tr].mean() - y[te]) ** 2).mean()))
        pooled_p.append(p[te]); pooled_y.append(y[te])
    P_, Y_ = np.concatenate(pooled_p), np.concatenate(pooled_y)
    edges = np.linspace(0, 1, BINS + 1); bi = np.clip(np.digitize(P_, edges[1:-1]), 0, BINS - 1)
    rel = [{"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "pictures": int((bi == b).sum()),
            "mean_p": round(float(P_[bi == b].mean()), 4) if (bi == b).any() else None,
            "share_scenes": round(float(Y_[bi == b].mean()), 4) if (bi == b).any() else None} for b in range(BINS)]
    out = {"experiment": "Calibration of the corrected design's probability of a scene, 200 book-disjoint redraws",
           "pictures": int(len(sp)), "draws": rb.N_DRAWS, "seed": rb.SEED, "bins": BINS,
           "per_score": {n: {k: rb.summarise(v) for k, v in d.items()} for n, d in acc.items()},
           "brier_of_base_rate": rb.summarise(base), "pooled_reliability_combined": rel}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for n, d in out["per_score"].items():
        print(f"{n:30s} ECE {d['ece']['mean']:.4f} [{d['ece']['p2.5']:.4f},{d['ece']['p97.5']:.4f}]  Brier {d['brier']['mean']:.4f}")
    print("Brier of base rate", out["brier_of_base_rate"]["mean"])
    for r in rel: print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
