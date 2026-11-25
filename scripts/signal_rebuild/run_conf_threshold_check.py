#!/usr/bin/env python3
"""AR2's last item: does scoring the detector down to confidence 0.001 instead of 0.01 change anything?

The object agent reads each class's highest box confidence. Boxes below the prediction threshold
are dropped before that max is taken, so a class with only weak evidence scores exactly zero. The
architecture track asks for 0.001 rather than 0.01, on the grounds that truncating weak evidence to
zero is part of why so many pictures looked empty. This measures whether it matters:

  how many (picture, class) cells were zero at 0.01 and are not at 0.001
  per-class detector AUROC on the 300 hand-annotated pictures, both thresholds
  the corrected + absence pipeline on the committed split, and over 200 book-disjoint redraws paired
  between the two thresholds

    python run_conf_threshold_check.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
CS = SR / "class_scores"
OUT = SR / "conf_threshold_check.json"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
FILES = {"0.01": CS / "new_student_class_scores.jsonl", "0.001": CS / "new_student_class_scores_conf0001.jsonl"}
W = np.array([0.6, 0.2, 0.2])
N_DRAWS, SEED = 200, 20260926


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    import tier1_signals as t1

    idx, nat, _old, scene_abs = e2e.native_vectors()
    C = t1.CLASSES
    obj = {}
    for thr, f in FILES.items():
        M = np.zeros((len(idx), len(C)))
        for line in open(f):
            r = json.loads(line); k = idx.get(r["image_id"])
            if k is not None:
                for c, v in r["max_conf"].items():
                    if c in C: M[k, C.index(c)] = v
        obj[thr] = M
    lo, hi = obj["0.001"], obj["0.01"]
    newly = (hi == 0) & (lo > 0)

    out = {"experiment": "Detector scored down to confidence 0.001 against 0.01",
           "cells": {"zero_at_0.01": int((hi == 0).sum()), "zero_at_0.001": int((lo == 0).sum()),
                     "zero_at_0.01_but_scored_at_0.001": int(newly.sum()),
                     "pictures_with_no_class_scored": {"0.01": int((hi.max(1) == 0).sum()),
                                                       "0.001": int((lo.max(1) == 0).sum())},
                     "per_class_newly_scored": {c: int(newly[:, i].sum()) for i, c in enumerate(C)}},
           "distinct_values": {thr: int(len(np.unique(np.round(M, 9)))) for thr, M in obj.items()}}

    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([idx[i] for i in truth.index])
    out["detector_auroc_on_300"] = {c: {thr: round(float(roc_auc_score(truth[c].astype(int).values, M[rows, i])), 4)
                                        for thr, M in obj.items()} for i, c in enumerate(C)}
    out["detector_auroc_on_300"]["mean"] = {thr: round(float(np.mean([out["detector_auroc_on_300"][c][thr] for c in C])), 4)
                                            for thr in obj}

    sp = pd.read_csv(SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    ra, y, books = sp.ppn_id.map(idx).values, sp.has_scene.values, sp.ppn.values
    def run(tr_mask, te_mask, O):
        V = {**nat, "scene": scene_abs, "object": O}
        P = np.column_stack([e2e.to_probability(V[k], ra[tr_mask], y[tr_mask], ra) for k in ("object", "scene", "vlm")])
        s = P @ W
        return roc_auc_score(y[te_mask], s[te_mask]), float((((s >= 0.5).astype(int)) != y)[te_mask].mean())

    tr, te = (sp.split == "train").values, (sp.split == "test").values
    out["committed_split"] = {thr: dict(zip(("scene_auroc", "test_error"), map(lambda x: round(x, 4), run(tr, te, M))))
                              for thr, M in obj.items()}

    rng = np.random.default_rng(SEED); ub = np.unique(books); dA, dE = [], []
    for _ in range(N_DRAWS):
        perm = rng.permutation(ub); b_tr = set(perm[:int(len(ub) * 0.6)]); b_te = set(perm[int(len(ub) * 0.8):])
        trm = np.array([b in b_tr for b in books]); tem = np.array([b in b_te for b in books])
        (a_lo, e_lo), (a_hi, e_hi) = run(trm, tem, lo), run(trm, tem, hi)
        dA.append(a_lo - a_hi); dE.append(e_hi - e_lo)
    f = lambda xs: {"mean": round(float(np.mean(xs)), 4), "p2.5": round(float(np.percentile(xs, 2.5)), 4),
                    "p97.5": round(float(np.percentile(xs, 97.5)), 4), "share_above_zero": round(float((np.array(xs) > 0).mean()), 4)}
    out["redraws_paired_0.001_minus_0.01"] = {"scene_auroc_gain": f(dA), "error_rate_reduction": f(dE)}

    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
