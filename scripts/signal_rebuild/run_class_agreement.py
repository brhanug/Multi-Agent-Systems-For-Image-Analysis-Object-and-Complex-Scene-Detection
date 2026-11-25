#!/usr/bin/env python3
"""Class-level agreement between the systems (C2, C3) and heterogeneity of the per-class signal (C5).

C2  agreement(m1, m2, c, i) = 1 - |s_m1(i,c) - s_m2(i,c)|, on the within-corpus percentile layer
    of the per-class score matrix (the raw scores are on different scales), for the three pairs
    of detector, SigLIP and LLaVA and the eight frozen classes, over all 12,110 pictures.
    Aggregated per class, per pair, per picture, and globally; the full tensor is stored.
C3  chance-corrected agreement on yes/no decisions. Each system's per-class cut-off is the Youden
    point on the 300 annotated pictures. Over the corpus: Cohen's kappa per pair and class,
    Fleiss' kappa across all three, prevalence, raw agreement and PABAK. On the 300: each system
    against the human rater, with the cut-off cross-fitted (five folds) so no picture is judged
    by a cut-off that saw it, beside the machine pairs on the same pictures.
C5  effective rank (perplexity of the covariance spectrum, as in the ensemble heterogeneity
    diagnostic) of the per-class tensor, per class across the three systems and over all 24
    system-class columns.

    python run_class_agreement.py [--verify]
"""
import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_curve
from sklearn.model_selection import StratifiedKFold

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/class_agreement/class_agreement.json"
TENSOR = BASE / "results/signal_rebuild/class_agreement/agreement_tensor.npz"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SYSTEMS = ["detector", "siglip", "llava"]
SEED = 20260919


def youden(v, y):
    fpr, tpr, th = roc_curve(y, v)
    return float(th[np.argmax(tpr - fpr)])


def fleiss(D):
    """Fleiss' kappa for binary ratings: D is pictures x raters of 0/1."""
    n = D.shape[1]; k1 = D.sum(1); k0 = n - k1
    P_i = (k1 * (k1 - 1) + k0 * (k0 - 1)) / (n * (n - 1))
    p1 = D.mean(); Pe = p1 ** 2 + (1 - p1) ** 2
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else None


def effective_rank(X):
    ev = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None)
    p = ev / ev.sum(); p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    ci = [mc.index(c) for c in CLASSES]; si = [ms.index(s) for s in SYSTEMS]
    P = m["percentile"][:, si][:, :, ci].astype(np.float64)      # pictures x systems x classes
    R = m["raw"][:, si][:, :, ci].astype(np.float64)
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index])
    pairs = list(combinations(range(len(SYSTEMS)), 2)); pname = [f"{SYSTEMS[x]}-{SYSTEMS[y]}" for x, y in pairs]

    # C2 -- continuous agreement on the percentile scale
    A = np.stack([1 - np.abs(P[:, x] - P[:, y]) for x, y in pairs], axis=1)   # pictures x pairs x classes
    per_picture = A.mean(axis=(1, 2))
    c2 = {"global_mean": round(float(A.mean()), 4),
          "per_pair": {p: round(float(A[:, k].mean()), 4) for k, p in enumerate(pname)},
          "per_class": {c: round(float(A[:, :, j].mean()), 4) for j, c in enumerate(CLASSES)},
          "per_pair_and_class": {p: {c: round(float(A[:, k, j].mean()), 4) for j, c in enumerate(CLASSES)} for k, p in enumerate(pname)},
          "per_picture_mean": {"p10": round(float(np.percentile(per_picture, 10)), 4), "median": round(float(np.median(per_picture)), 4),
                               "p90": round(float(np.percentile(per_picture, 90)), 4)},
          "chance_level_note": "two independent uniform percentiles agree 0.667 on average"}

    # C3 -- decisions: Youden cut-offs on the 300, applied corpus-wide
    D = np.zeros_like(R, dtype=int)
    for s in range(len(SYSTEMS)):
        for j, c in enumerate(CLASSES):
            D[:, s, j] = (R[:, s, j] >= youden(R[rows, s, j], truth[c].values)).astype(int)
    c3 = {}
    for j, c in enumerate(CLASSES):
        e = {"prevalence": {SYSTEMS[s]: round(float(D[:, s, j].mean()), 4) for s in range(len(SYSTEMS))},
             "fleiss_kappa": round(fleiss(D[:, :, j]), 4), "pairs": {}}
        for k, (x, y) in enumerate(pairs):
            po = float((D[:, x, j] == D[:, y, j]).mean())
            e["pairs"][pname[k]] = {"cohen_kappa": round(float(cohen_kappa_score(D[:, x, j], D[:, y, j])), 4),
                                    "raw_agreement": round(po, 4), "pabak": round(2 * po - 1, 4)}
        c3[c] = e

    # C3 on the 300 -- each system against the human rater (cross-fitted cut-offs) and the machine pairs
    human = {}
    for j, c in enumerate(CLASSES):
        y = truth[c].values; Dh = np.zeros((len(rows), len(SYSTEMS)), dtype=int)
        for s in range(len(SYSTEMS)):
            v = R[rows, s, j]
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(v, y):
                Dh[te, s] = (v[te] >= youden(v[tr], y[tr])).astype(int)
        human[c] = {"human_positives": int(y.sum()),
                    "system_vs_human_kappa": {SYSTEMS[s]: round(float(cohen_kappa_score(Dh[:, s], y)), 4) for s in range(len(SYSTEMS))},
                    "machine_pair_kappa": {pname[k]: round(float(cohen_kappa_score(Dh[:, x], Dh[:, y])), 4) for k, (x, y) in enumerate(pairs)}}

    # C5 -- effective rank of the per-class signal
    c5 = {"per_class_across_three_systems": {c: round(effective_rank(P[:, :, j]), 3) for j, c in enumerate(CLASSES)},
          "per_class_max": 3,
          "all_24_columns": round(effective_rank(P.reshape(len(P), -1)), 3), "all_max": len(SYSTEMS) * len(CLASSES),
          "scalar_signal_for_reference": {"colibri": 2.498, "europeana": 1.367, "max": 4}}
    c5["per_class_mean"] = round(float(np.mean(list(c5["per_class_across_three_systems"].values()))), 3)

    out = {"experiment": "Class-level agreement (C2, C3) and heterogeneity of the per-class signal (C5)",
           "pictures": len(ids), "classes": CLASSES, "systems": SYSTEMS, "c2_agreement": c2, "c3_kappa_corpus": c3,
           "c3_on_300_vs_human": human, "c5_effective_rank": c5}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(TENSOR, ids=np.array(ids), pairs=np.array(pname), classes=np.array(CLASSES), agreement=A.astype(np.float32))
    print("C2 global", c2["global_mean"], "| per pair", c2["per_pair"], "\n   per class", c2["per_class"])
    print("\nC3 corpus: Fleiss kappa and pairwise Cohen kappa per class")
    for c, e in c3.items():
        print(f"  {c:9s} Fleiss {e['fleiss_kappa']:+.2f}  " + "  ".join(f"{p} {v['cohen_kappa']:+.2f} (PABAK {v['pabak']:+.2f})" for p, v in e["pairs"].items()))
    print("\nC3 on the 300: system vs human | machine pairs")
    for c, e in human.items():
        print(f"  {c:9s} " + "  ".join(f"{s} {v:+.2f}" for s, v in e["system_vs_human_kappa"].items()) + "  |  " + "  ".join(f"{p} {v:+.2f}" for p, v in e["machine_pair_kappa"].items()))
    print("\nC5 effective rank per class (of 3):", c5["per_class_across_three_systems"], "| mean", c5["per_class_mean"], "| all 24 columns:", c5["all_24_columns"])


if __name__ == "__main__":
    raise SystemExit(main())
