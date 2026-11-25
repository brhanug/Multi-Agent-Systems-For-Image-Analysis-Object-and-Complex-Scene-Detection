#!/usr/bin/env python3
"""What the missed-object promise of Task O means in practice (extra analysis of the planned run's lists).

The promise bounds the mean over pictures of the share of a picture's present classes left off its
list; pictures with no object count as 0. This repeats Task O's 200 redraws (combined design, same
seed, fit/calibration/test) and reports, for the same lists, the mean share missed among pictures that
contain at least one object, and the share of all present objects (pooled) that are missed.

    python run_object_promise_detail.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_object_classes as ro

OUT = rb.SR / "object_promise_detail.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    R, pos = ro.scores()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books); Rr = R[rows]
    keep = {al: {"promised_measure": [], "among_pictures_with_objects": [], "pooled_objects": []} for al in ro.ALPHAS}
    rng = np.random.default_rng(ro.SEED)
    for _ in range(ro.N_DRAWS):
        perm = rng.permutation(uniq); n1, n2 = int(len(uniq) * 0.4), int(len(uniq) * 0.3)
        bf, bc = set(perm[:n1]), set(perm[n1:n1 + n2])
        fit = np.array([b in bf for b in books]); cal = np.array([b in bc for b in books]); te = ~(fit | cal)
        P = ro.mapped(Rr, Y, np.where(fit)[0], np.arange(len(Rr))).mean(1)
        for al in ro.ALPHAS:
            t = ro.threshold(P[cal], Y[cal], al); S = P[te] >= t; Yt = Y[te]; f = ro.fnr(S, Yt); has = Yt.sum(1) > 0
            keep[al]["promised_measure"].append(float(f.mean()))
            keep[al]["among_pictures_with_objects"].append(float(f[has].mean()))
            keep[al]["pooled_objects"].append(float(((Yt == 1) & ~S).sum() / max(1, Yt.sum())))
    out = {"extra": "added after Task O (class_tasks/PLAN.md, commit d9a7fa1); same lists",
           "pictures_without_objects": int((Y.sum(1) == 0).sum()),
           "results": {f"alpha {al}": {m: rb.summarise(v) for m, v in z.items()} for al, z in keep.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for al, z in out["results"].items(): print(al, {m: v["mean"] for m, v in z.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
