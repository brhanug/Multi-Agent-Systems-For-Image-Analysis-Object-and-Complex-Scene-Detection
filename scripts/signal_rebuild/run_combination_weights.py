#!/usr/bin/env python3
"""E6 (self_annotation_package/more_experiments/PLAN_combination.md, commit e9b630f): fixed or learned
combination weights for the scene classes and the object classes.

Variants: fixed (0.6/0.2/0.2, or the mean for objects), equal weights, learned weights by stacking on
out-of-fold agent probabilities inside the training books, one joint mapping on all reports, describer
alone. Same redraws and measures as Tasks S and O.

    python run_combination_weights.py [--verify]
"""
import argparse, json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import run_best_config_repeated as rb
import run_scene_classes as rs
import run_object_classes as ro
from run_more_experiments import redraws

OUT = rb.SR / "combination_weights.json"


def folds(books, k=5, seed=7):
    ub = np.unique(books); rng = np.random.default_rng(seed); perm = rng.permutation(ub)
    return [np.isin(books, perm[i::k]) for i in range(k)]


def meta_multi(X, y, Xall, K):
    m = LogisticRegression(max_iter=5000, C=1.0).fit(X, y)
    P = np.zeros((len(Xall), K)); P[:, m.classes_] = m.predict_proba(Xall); return P


def scene_part():
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({a: i for i, a in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    books = d.ppn.values; K = rs.K; ags = ("object", "scene", "vlm")
    names = ["fixed 0.6/0.2/0.2", "equal weights", "learned weights (stacking)", "one joint mapping", "describer alone"]
    acc = {n: {"macro_f1": [], "balanced_accuracy": [], "approved@0.05": [], "wrong_among_approved@0.05": []} for n in names}
    learned_w = []
    for tr, cal, te in redraws(books):
        Pk = {k: rs.to_probs(V[k], rows[tr], y[tr], rows) for k in ags}
        tri = np.where(tr)[0]; oof = np.zeros((len(tri), 3 * K))
        for f in folds(books[tri]):
            a, b = tri[~f], tri[f]
            oof[f] = np.column_stack([rs.to_probs(V[k], rows[a], y[a], rows[b]) for k in ags])
        stack_all = np.column_stack([Pk[k] for k in ags])
        mm = LogisticRegression(max_iter=5000, C=1.0).fit(oof, y[tri])
        Pst = np.zeros((len(rows), K)); Pst[:, mm.classes_] = mm.predict_proba(stack_all)
        # summary of how much each agent counts in the learned combination: share of absolute coefficients
        c = np.abs(mm.coef_).reshape(len(mm.classes_), 3, K).sum(axis=(0, 2)); learned_w.append(c / c.sum())
        Pj = rs.to_probs(np.column_stack([V[k] for k in ags]), rows[tr], y[tr], rows)
        Ps = {names[0]: 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"], names[1]: sum(Pk.values()) / 3,
              names[2]: Pst, names[3]: Pj, names[4]: Pk["vlm"]}
        for n, P in Ps.items():
            pred = P.argmax(1); yt = y[te]; k_ = acc[n]
            k_["macro_f1"].append(float(f1_score(yt, pred[te], labels=range(K), average="macro", zero_division=0)))
            k_["balanced_accuracy"].append(float(np.mean([(pred[te][yt == j] == j).mean() for j in range(K) if (yt == j).any()])))
            u = 1 - P.max(1); e = (pred != y).astype(int); lam = rs.lam_top1(u[cal], e[cal], 0.05); n_ = cal.sum()
            feas = (n_ / (n_ + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (n_ + 1.0) <= 0.05
            ap = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
            k_["approved@0.05"].append(float(ap.mean())); k_["wrong_among_approved@0.05"].append(float(e[te][ap].mean()) if ap.any() else 0.0)
    return {"pictures": int(len(d)),
            "variants": {n: {m: rb.summarise(v) for m, v in k.items()} for n, k in acc.items()},
            "minus_fixed": {n: {m: rb.paired(np.array(acc[n][m]) - np.array(acc[names[0]][m])) for m in ("macro_f1", "balanced_accuracy", "approved@0.05")}
                            for n in names[1:]},
            "learned_agent_shares": dict(zip(ags, [round(float(x), 3) for x in np.mean(learned_w, 0)])),
            "extra_minus_describer_alone": {n: {m: rb.paired(np.array(acc[n][m]) - np.array(acc[names[4]][m])) for m in ("macro_f1", "balanced_accuracy", "approved@0.05")}
                                            for n in (names[2], names[3])}}   # added after the planned run


def object_part():
    R, pos = ro.scores()
    import pandas as pd
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books); Rr = R[rows]; C = len(ro.CLASSES)
    names = ["fixed mean", "learned weights (stacking)", "one joint mapping", "describer alone"]
    acc = {n: {"macro_auroc": [], "macro_ap": [], "miss@0.1": [], "listed@0.1": [], "wrong_per_picture@0.1": []} for n in names}
    rng = np.random.default_rng(ro.SEED)
    for _ in range(ro.N_DRAWS):
        perm = rng.permutation(uniq); n1, n2 = int(len(uniq) * 0.4), int(len(uniq) * 0.3)
        bf, bc = set(perm[:n1]), set(perm[n1:n1 + n2])
        fit = np.array([b in bf for b in books]); cal = np.array([b in bc for b in books]); te = ~(fit | cal)
        fi = np.where(fit)[0]
        P3 = ro.mapped(Rr, Y, fi, np.arange(len(Rr)))
        oof = np.zeros((len(fi), 3, C))
        for f in folds(books[fi]):
            oof[f] = ro.mapped(Rr, Y, fi[~f], fi[f])
        Pst = np.zeros((len(Rr), C)); Pj = np.zeros((len(Rr), C))
        for c in range(C):
            yc = Y[fi, c]
            if 0 < yc.sum() < len(yc):
                Pst[:, c] = LogisticRegression(C=1.0, max_iter=5000).fit(oof[:, :, c], yc).predict_proba(P3[:, :, c])[:, 1]
                Xj = ro.logit(Rr[:, :, c]); Pj[:, c] = LogisticRegression(C=1.0, max_iter=5000).fit(Xj[fi], yc).predict_proba(Xj)[:, 1]
            else:
                Pst[:, c] = Pj[:, c] = float(yc.mean())
        Ps = {names[0]: P3.mean(1), names[1]: Pst, names[2]: Pj, names[3]: P3[:, 2]}
        for n, P in Ps.items():
            au, apv = [], []
            for c in range(C):
                yt = Y[te, c]
                if 0 < yt.sum() < len(yt): au.append(roc_auc_score(yt, P[te, c])); apv.append(average_precision_score(yt, P[te, c]))
            k_ = acc[n]; k_["macro_auroc"].append(float(np.mean(au))); k_["macro_ap"].append(float(np.mean(apv)))
            t = ro.threshold(P[cal], Y[cal], 0.10); S = P[te] >= t
            k_["miss@0.1"].append(float(ro.fnr(S, Y[te]).mean())); k_["listed@0.1"].append(float(S.sum(1).mean()))
            k_["wrong_per_picture@0.1"].append(float(((Y[te] == 0) & S).sum(1).mean()))
    return {"pictures": int(len(rows)),
            "variants": {n: {m: rb.summarise(v) for m, v in k.items()} for n, k in acc.items()},
            "minus_fixed": {n: {m: rb.paired(np.array(acc[n][m]) - np.array(acc[names[0]][m])) for m in ("macro_auroc", "macro_ap", "listed@0.1")}
                            for n in names[1:]}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    out = {"plan": "self_annotation_package/more_experiments/PLAN_combination.md (commit e9b630f)",
           "scene_classes": scene_part(), "object_classes": object_part()}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for part in ("scene_classes", "object_classes"):
        print("==", part)
        for n, v in out[part]["variants"].items(): print(f"  {n:28s}", {m: x["mean"] for m, x in v.items()})
        for n, v in out[part]["minus_fixed"].items(): print(f"  {n:28s} minus fixed", {m: (x["mean"], x["p2.5"], x["p97.5"]) for m, x in v.items()})
    print("learned agent shares (scene classes):", out["scene_classes"]["learned_agent_shares"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
