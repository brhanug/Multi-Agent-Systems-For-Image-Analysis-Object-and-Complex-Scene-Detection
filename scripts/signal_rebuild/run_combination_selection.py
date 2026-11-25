#!/usr/bin/env python3
"""S1 and S2 of self_annotation_package/more_experiments/PLAN_selection.md (commit 42db659).

S1: the scene-class combination chosen inside each redraw by five-fold book cross-validation on the
training books (fixed, equal, learned weights, joint mapping), then scored on the test books.
S2: the joint mapping at C = 0.1 and C = 10 (sensitivity only).

    python run_combination_selection.py [--verify]
"""
import argparse, json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import run_best_config_repeated as rb
import run_scene_classes as rs
from run_more_experiments import redraws
from run_combination_weights import folds

OUT = rb.SR / "combination_selection.json"
K, AGS = rs.K, ("object", "scene", "vlm")
NAMES = ["fixed 0.6/0.2/0.2", "equal weights", "learned weights (stacking)", "one joint mapping"]


def fit_probs(X, rf, yf, rall, C=1.0):
    m = LogisticRegression(max_iter=5000, C=C).fit(X[rf], yf)
    P = np.zeros((len(rall), K)); P[:, m.classes_] = m.predict_proba(X[rall]); return P


def variants(V, Xj, rows, y, fit_idx, books, eval_idx):
    """Probabilities of the four variants for eval_idx, everything fitted on fit_idx only."""
    rf, yf, re = rows[fit_idx], y[fit_idx], rows[eval_idx]
    Pk = {k: rs.to_probs(V[k], rf, yf, re) for k in AGS}
    oof = np.zeros((len(fit_idx), 3 * K))
    for f in folds(books[fit_idx]):
        a, b = fit_idx[~f], fit_idx[f]
        oof[f] = np.column_stack([rs.to_probs(V[k], rows[a], y[a], rows[b]) for k in AGS])
    mm = LogisticRegression(max_iter=5000, C=1.0).fit(oof, yf)
    Pst = np.zeros((len(eval_idx), K)); Pst[:, mm.classes_] = mm.predict_proba(np.column_stack([Pk[k] for k in AGS]))
    return {NAMES[0]: 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"], NAMES[1]: sum(Pk.values()) / 3,
            NAMES[2]: Pst, NAMES[3]: fit_probs(Xj, rf, yf, re)}


def bal_acc(yt, pt):
    return float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    Xj = np.column_stack([V[k] for k in AGS])
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    books = d.ppn.values
    s1 = {"balanced_accuracy": [], "macro_f1": [], "fixed_balanced_accuracy": [], "chosen": []}
    s2 = {c: {"balanced_accuracy": [], "macro_f1": [], "approved@0.05": []} for c in (0.1, 1.0, 10.0)}
    for tr, cal, te in redraws(books):
        tri, tei = np.where(tr)[0], np.where(te)[0]
        # S1: inner five-fold choice on the training books
        score = {n: [] for n in NAMES}
        for f in folds(books[tri], seed=11):
            fi, ev = tri[~f], tri[f]
            for n, P in variants(V, Xj, rows, y, fi, books, ev).items():
                score[n].append(bal_acc(y[ev], P.argmax(1)))
        best = max(NAMES, key=lambda n: np.mean(score[n]))
        Pall = variants(V, Xj, rows, y, tri, books, tei)
        pt = Pall[best].argmax(1); yt = y[tei]
        s1["balanced_accuracy"].append(bal_acc(yt, pt)); s1["chosen"].append(best)
        s1["macro_f1"].append(float(f1_score(yt, pt, labels=range(K), average="macro", zero_division=0)))
        s1["fixed_balanced_accuracy"].append(bal_acc(yt, Pall[NAMES[0]].argmax(1)))
        # S2: regularisation of the joint mapping
        for C in s2:
            P = fit_probs(Xj, rows[tr], y[tr], rows, C); pred = P.argmax(1)
            s2[C]["balanced_accuracy"].append(bal_acc(y[te], pred[te]))
            s2[C]["macro_f1"].append(float(f1_score(y[te], pred[te], labels=range(K), average="macro", zero_division=0)))
            u = 1 - P.max(1); e = (pred != y).astype(int); lam = rs.lam_top1(u[cal], e[cal], 0.05); n_ = cal.sum()
            feas = (n_ / (n_ + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (n_ + 1.0) <= 0.05
            s2[C]["approved@0.05"].append(float(((u[te] < lam) if feas else np.zeros(te.sum(), bool)).mean()))
    diff = np.array(s1["balanced_accuracy"]) - np.array(s1["fixed_balanced_accuracy"])
    out = {"plan": "self_annotation_package/more_experiments/PLAN_selection.md (commit 42db659)", "pictures": int(len(d)),
           "S1_nested_choice": {"balanced_accuracy": rb.summarise(s1["balanced_accuracy"]), "macro_f1": rb.summarise(s1["macro_f1"]),
                                "minus_fixed_balanced_accuracy": rb.paired(diff),
                                "times_chosen": {n: int(sum(c == n for c in s1["chosen"])) for n in NAMES}},
           "S2_joint_mapping_C": {str(C): {m: rb.summarise(v) for m, v in z.items()} for C, z in s2.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
