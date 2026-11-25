#!/usr/bin/env python3
"""Agreement between the systems on calibrated scores (decision 2, signed off by Prof. Mandl).

The first agreement figures (run_class_agreement.py, run_scene_agreement.py) used within-archive
ranks, because the systems' raw scores are on different scales. Here every system's score is first
calibrated to a probability that the class is present -- Platt scaling on the logit of the raw
score, per system and class, fitted on the 300 hand-labelled pictures -- and agreement is then

    agreement(m1, m2, c, i) = 1 - |p_m1(i, c) - p_m2(i, c)|

Objects: detector, SigLIP, LLaVA on the eight frozen classes. Scenes: SigLIP and LLaVA on the five
scene types. Over all 12,110 pictures:

  agreement    mean per pair and class, beside what two systems with the same calibrated scores
               would reach if they were unrelated (one system's column shuffled across pictures),
               so the excess over that is the agreement that is not just shared base rates
  decisions    yes where p >= 0.5: Cohen's kappa per pair and class, prevalence

On the 300 (calibrators cross-fitted over five folds, so no picture is scored by a fit that saw it):
Brier score of each system's calibrated probability, each system against the rater, and the machine
pair, as Cohen's kappa at p >= 0.5.

    python run_calibrated_agreement.py [--verify]
"""
import argparse, glob, json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
CS = SR / "class_scores"
OUT = SR / "class_agreement/calibrated_agreement.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
SEED = 20261011


def norm(i):
    i = str(i)
    return i if "/" in i else i.replace("_", "/", 1)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def platt(x_fit, y_fit, x_all):
    if not 0 < y_fit.sum() < len(y_fit):
        return np.full(len(x_all), float(y_fit.mean()))
    return LogisticRegression(C=100.0).fit(logit(x_fit).reshape(-1, 1), y_fit).predict_proba(logit(x_all).reshape(-1, 1))[:, 1]


def kappa(a, b):
    return None if len(set(a)) < 2 and len(set(b)) < 2 else round(float(cohen_kappa_score(a, b)), 4)


def block(R, rows, Y, systems, classes):
    """R: pictures x systems x classes raw scores; rows: the 300 in R; Y: 300 x classes truth."""
    n, S, C = R.shape
    P = np.zeros_like(R); Pcv = np.zeros((len(rows), S, C))
    for s in range(S):
        for c in range(C):
            y = Y[:, c]; P[:, s, c] = platt(R[rows, s, c], y, R[:, s, c])
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y):
                Pcv[te, s, c] = platt(R[rows[tr], s, c], y[tr], R[rows[te], s, c])
    pairs = list(combinations(range(S), 2)); pname = [f"{systems[a]}-{systems[b]}" for a, b in pairs]
    rng = np.random.default_rng(SEED); perm = rng.permutation(n)
    corpus = {}
    for (a, b), p in zip(pairs, pname):
        per = {}
        for c, cname in enumerate(classes):
            agree = float((1 - np.abs(P[:, a, c] - P[:, b, c])).mean())
            unrelated = float((1 - np.abs(P[:, a, c] - P[perm, b, c])).mean())
            da, db = (P[:, a, c] >= 0.5).astype(int), (P[:, b, c] >= 0.5).astype(int)
            per[cname] = {"agreement": round(agree, 4), "if_unrelated": round(unrelated, 4), "excess": round(agree - unrelated, 4),
                          "kappa_at_0.5": kappa(da, db), "prevalence": [round(float(da.mean()), 4), round(float(db.mean()), 4)]}
        corpus[p] = {"per_class": per,
                     "mean_agreement": round(float(np.mean([v["agreement"] for v in per.values()])), 4),
                     "mean_if_unrelated": round(float(np.mean([v["if_unrelated"] for v in per.values()])), 4),
                     "mean_excess": round(float(np.mean([v["excess"] for v in per.values()])), 4)}
    on300 = {}
    for c, cname in enumerate(classes):
        y = Y[:, c]; D = (Pcv[:, :, c] >= 0.5).astype(int)
        on300[cname] = {"positives": int(y.sum()),
                        "brier": {systems[s]: round(float(((Pcv[:, s, c] - y) ** 2).mean()), 4) for s in range(S)},
                        "brier_of_base_rate": round(float(y.mean() * (1 - y.mean())), 4),
                        "system_vs_rater_kappa": {systems[s]: kappa(D[:, s], y) for s in range(S)},
                        "machine_pair_kappa": {p: kappa(D[:, a], D[:, b]) for (a, b), p in zip(pairs, pname)}}
    return {"pictures": int(n), "corpus": corpus,
            "global_mean_agreement": round(float(np.mean([v["mean_agreement"] for v in corpus.values()])), 4),
            "global_mean_if_unrelated": round(float(np.mean([v["mean_if_unrelated"] for v in corpus.values()])), 4),
            "on_300_cross_fitted": on300}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    sys_o = ["detector", "siglip", "llava"]
    R = np.nan_to_num(m["raw"][:, [ms.index(s) for s in sys_o]][:, :, [mc.index(c) for c in CLASSES]].astype(np.float64))
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index])
    objects = block(R, rows, truth[CLASSES].astype(int).values, sys_o, CLASSES)

    sig, lla = {}, {}
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sig[norm(r["image_id"])] = [float(r["sigmoid"].get(s, 0.0)) for s in SCENES]
    for f in sorted(glob.glob(str(CS / "llava_scene_scores/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); lla[norm(r["image_id"])] = [float(r["p_yes"][s]) for s in SCENES]
    sids = sorted(set(sig) & set(lla)); spos = {i: k for k, i in enumerate(sids)}
    Rs = np.stack([np.array([sig[i] for i in sids]), np.array([lla[i] for i in sids])], axis=1)
    t = truth[truth.index.isin(spos)]
    Ys = np.column_stack([(t.scene.str.strip().str.lower() == s).astype(int).values for s in SCENES])
    scenes = block(Rs, np.array([spos[i] for i in t.index]), Ys, ["siglip", "llava"], SCENES)

    out = {"experiment": "Agreement between systems on Platt-calibrated probabilities (objects and scenes)",
           "calibration": "Platt scaling on the logit of the raw score, per system and class, fitted on the 300 hand-labelled pictures; cross-fitted (5 folds) for every figure on the 300",
           "objects": objects, "scenes": scenes}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for name, b in (("objects", objects), ("scenes", scenes)):
        print(f"\n{name}: {b['pictures']} pictures; mean agreement {b['global_mean_agreement']} (if unrelated {b['global_mean_if_unrelated']})")
        for p, v in b["corpus"].items():
            print(f"  {p}: {v['mean_agreement']} (unrelated {v['mean_if_unrelated']}, excess {v['mean_excess']})")
            print("     " + "  ".join(f"{c} {x['agreement']:.2f}/{x['if_unrelated']:.2f} k={x['kappa_at_0.5']}" for c, x in v["per_class"].items()))
        print("  on the 300 (cross-fitted): class, positives, Brier per system (base rate), kappa vs rater, machine pairs")
        for c, v in b["on_300_cross_fitted"].items():
            print(f"     {c:9s} {v['positives']:3d}  {v['brier']} ({v['brier_of_base_rate']})  {v['system_vs_rater_kappa']}  {v['machine_pair_kappa']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
