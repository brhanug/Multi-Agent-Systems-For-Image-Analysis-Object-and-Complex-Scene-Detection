#!/usr/bin/env python3
"""Per-category human agreement on scene labels, with intervals (Phase 1, A4).

The single categorical kappa (0.3367 on the 124 genuinely double-reviewed pictures,
see compute_real_interannotator_kappa.py) cannot say which categories raters agree
on. This takes the same 124 pictures and, for each category one-vs-rest, reports
raw agreement, Cohen's kappa with a bootstrap interval, each rater's positive count,
prevalence, bias, and PABAK -- the prevalence-adjusted kappa that stays meaningful
when a category is rare, where plain kappa collapses (the kappa paradox).

    python run_per_class_human_kappa.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path("/data/brhanu/thesis_project")
C1_CSV = BASE / "human_spatial_audit/user_annotations_800.csv"
C2_CSV = BASE / "human_spatial_audit/user_annotations_300.csv"
OUT = BASE / "results/multi_agent/per_class_human_kappa.json"
REAL_REVIEW_CUTOFF, B, SEED = 800, 4000, 20260916
USABLE_KAPPA = 0.4


def pairs():
    c1 = pd.read_csv(C1_CSV); c2 = pd.read_csv(C2_CSV)
    c1["cvat_id"] = c1.index
    c1["full_id"] = c1["raw_name"].str.replace("images/", "", regex=False).str.replace(".jpg", "", regex=False)
    m = c2.merge(c1, left_on="image_id", right_on="full_id", how="inner", suffixes=("_c2", "_c1"))
    m = m[m["cvat_id"] <= REAL_REVIEW_CUTOFF]
    return m["scene_labels"].fillna("none").str.strip().str.lower().values, m["scene"].fillna("none").str.strip().str.lower().values


def kappa(a, b):
    return None if len(set(a)) < 2 and len(set(b)) < 2 else float(cohen_kappa_score(a, b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); args = ap.parse_args()
    ra, rb = pairs()
    n = len(ra); rng = np.random.default_rng(SEED)
    overall = float(cohen_kappa_score(ra, rb, labels=sorted(set(ra) | set(rb))))
    idx = [rng.integers(0, n, n) for _ in range(B)]
    boot = [cohen_kappa_score(ra[i], rb[i], labels=sorted(set(ra) | set(rb))) for i in idx]
    per = []
    for c in sorted(set(ra) | set(rb)):
        a, b = (ra == c).astype(int), (rb == c).astype(int)
        po = float((a == b).mean()); k = kappa(a, b)
        row = {"category": c, "rater_a_positives": int(a.sum()), "rater_b_positives": int(b.sum()),
               "raw_agreement": round(po, 4), "prevalence": round(float((a.mean() + b.mean()) / 2), 4),
               "bias": round(float(abs(a.mean() - b.mean())), 4), "pabak": round(2 * po - 1, 4),
               "cohens_kappa": None if k is None else round(k, 4)}
        if a.sum() == 0 or b.sum() == 0:
            row.update(ci95=None, usable=False,
                       note="one rater never used this category, so kappa is 0 or undefined by construction")
        else:
            bs = [kappa(a[i], b[i]) for i in idx]
            bs = [x for x in bs if x is not None and not np.isnan(x)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update(ci95=[round(float(lo), 4), round(float(hi), 4)], usable=bool(k is not None and k >= USABLE_KAPPA))
        per.append(row)
    out = {"experiment": "Per-category human agreement on scene labels (A4)", "pictures": n,
           "rater_a": "CVAT expert scene review (cvat_id <= 800)", "rater_b": "Label Studio spatial-audit pass",
           "categorical_kappa": round(overall, 4),
           "categorical_kappa_ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
           "usable_threshold": USABLE_KAPPA, "per_category": per,
           "reading": "A category whose kappa stays below the threshold cannot carry a per-class result against these labels."}
    if args.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(f"{n} pictures | categorical kappa {out['categorical_kappa']} {out['categorical_kappa_ci95']}")
    for r in per: print(" ", r)


if __name__ == "__main__":
    raise SystemExit(main())
