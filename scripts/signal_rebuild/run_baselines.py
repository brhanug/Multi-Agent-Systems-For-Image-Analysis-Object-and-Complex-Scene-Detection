#!/usr/bin/env python3
"""Baselines for the three tasks: self_annotation_package/more_experiments/PLAN_baselines.md (commit 08e2412).

Scene classes: the deployed design's own answer, the describer zero-shot, a SigLIP linear probe.
Objects: the deployed detector alone, a SigLIP linear probe. Scene presence: a SigLIP linear probe.
Each on the pictures, labels and redraws of the system's own results; paired with the system.

    python run_baselines.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import run_best_config_repeated as rb
import run_scene_classes as rs
import run_object_classes as ro
from run_more_experiments import redraws

OUT = rb.SR / "baselines.json"
CS_GRID = [0.1, 1.0, 10.0]
K = rs.K


def embeddings(idx):
    z = np.load(rb.SR / "siglip_embeddings.npz", allow_pickle=True)
    pos = {str(i): k for k, i in enumerate(z["ids"])}
    E = z["vecs"].astype(np.float64); E /= np.linalg.norm(E, axis=1, keepdims=True)
    out = np.zeros((len(idx), E.shape[1]))
    for pid, r in idx.items():
        out[r] = E[pos[pid]]
    return out


def probe(X, y, groups, Xall, multi, score):
    """Logistic regression with C chosen by three-fold group cross-validation on (X, y)."""
    best, bc = -np.inf, 1.0
    for C in CS_GRID:
        vals = []
        for a, b in GroupKFold(3).split(X, y, groups):
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                continue
            m = LogisticRegression(max_iter=3000, C=C).fit(X[a], y[a])
            vals.append(score(y[b], m, X[b]))
        if vals and np.mean(vals) > best:
            best, bc = np.mean(vals), C
    m = LogisticRegression(max_iter=3000, C=bc).fit(X, y)
    if multi:
        P = np.zeros((len(Xall), K)); P[:, m.classes_] = m.predict_proba(Xall); return P, bc
    return m.predict_proba(Xall)[:, 1], bc


def bal(yt, pt):
    return float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any()]))


def crc_top1(P, y, cal, te, alpha=0.05):
    pred = P.argmax(1); u = 1 - P.max(1); e = (pred != y).astype(int); n = cal.sum()
    lam = rs.lam_top1(u[cal], e[cal], alpha)
    feas = (n / (n + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (n + 1.0) <= alpha
    auto = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
    return float(auto.mean()), (float(e[te][auto].mean()) if auto.any() else 0.0)


# ---------------------------------------------------------------- scene classes
def scene_draw(tr, cal, te, Xj, Erow, rows, y, books, fixed):
    out = {}
    Pj = rs.to_probs(Xj, rows[tr], y[tr], rows)
    sc_bal = lambda yy, m, X: bal(yy, np.array(m.classes_)[m.predict_proba(X).argmax(1)])
    Pp, C = probe(Erow[tr], y[tr], books[tr], Erow, True, sc_bal)
    preds = {"system": Pj.argmax(1), "linear probe": Pp.argmax(1), **fixed}
    yt = y[te]
    for name, pr in preds.items():
        pt = pr[te]
        out[name] = {"balanced_accuracy": bal(yt, pt), "accuracy": float((pt == yt).mean()),
                     "macro_f1": float(f1_score(yt, pt, labels=range(K), average="macro", zero_division=0))}
    for name, P in (("system", Pj), ("linear probe", Pp)):
        ap, w = crc_top1(P, y, cal, te); out[name]["approved@0.05"] = ap; out[name]["wrong_among_approved@0.05"] = w
    out["linear probe"]["C"] = C
    return out


def scene_part(idx, V, E):
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values; books = d.ppn.values
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[d.ppn_id]
    saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    sm = {}
    for line in open(rb.SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sm[r["image_id"].replace("_", "/", 1)] = r["argmax"]
    forced = np.array([rs.ANSWERS.index(sm[p]) for p in d.ppn_id])
    none = rs.ANSWERS.index("none")
    old = np.where(saa >= 0.5, forced, none)
    ls = rs.llava_scene(idx)[rows]; zs = np.where(ls.max(1) >= 0.5, ls.argmax(1), none)
    fixed = {"old system (deployed)": old, "describer zero-shot": zs}
    draws = list(redraws(books))
    res = Parallel(n_jobs=16)(delayed(scene_draw)(tr, cal, te, Xj, E[rows], rows, y, books, fixed) for tr, cal, te in draws)
    names = list(res[0])
    agg = {n: {m: rb.summarise([r[n][m] for r in res]) for m in res[0][n] if m != "C"} for n in names}
    agg["linear probe"]["C_chosen"] = {str(c): int(sum(r["linear probe"]["C"] == c for r in res)) for c in CS_GRID}
    paired = {n: {m: rb.paired(np.array([r["system"][m] - r[n][m] for r in res])) for m in ("balanced_accuracy", "macro_f1")}
              for n in names if n != "system"}
    return {"pictures": int(len(d)), "per_system": agg, "system_minus": paired,
            "old_system_never_says": [rs.ANSWERS[k] for k in range(K) if not (old == k).any()]}


# ---------------------------------------------------------------- objects
def object_draw(fit, cal, te, Rr, Y, Erow, books, old):
    C = len(ro.CLASSES); P3 = ro.mapped(Rr, Y, np.where(fit)[0], np.arange(len(Rr))).mean(1)
    Pp = np.zeros_like(P3); cs = []
    auc = lambda yy, m, X: roc_auc_score(yy, m.predict_proba(X)[:, 1])
    for c in range(C):
        yc = Y[fit, c]
        if 0 < yc.sum() < len(yc):
            Pp[:, c], bc = probe(Erow[fit], yc, books[fit], Erow, False, auc); cs.append(bc)
        else:
            Pp[:, c] = float(yc.mean())
    out = {}
    for name, P in (("system", P3), ("linear probe", Pp), ("old detector (deployed)", old)):
        au = [roc_auc_score(Y[te, c], P[te, c]) for c in range(C) if 0 < Y[te, c].sum() < te.sum()]
        out[name] = {"mean_auroc": float(np.mean(au))}
    for name, P in (("system", P3), ("linear probe", Pp)):
        t = ro.threshold(P[cal], Y[cal], 0.10); S = P[te] >= t
        out[name]["listed@0.1"] = float(S.sum(1).mean()); out[name]["missed_per_picture@0.1"] = float(ro.fnr(S, Y[te]).mean())
    return out


def object_part(E_by_id):
    R, pos = ro.scores()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books)
    m = np.load(ro.CS / "class_score_matrix.npz", allow_pickle=True)
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    old = np.nan_to_num(m["raw"][:, ms.index("detector")][:, [mc.index(c) for c in ro.CLASSES]].astype(float))[rows]
    Erow = np.array([E_by_id[i] for i in truth.index])
    rng = np.random.default_rng(ro.SEED); draws = []
    for _ in range(ro.N_DRAWS):
        perm = rng.permutation(uniq); n1, n2 = int(len(uniq) * 0.4), int(len(uniq) * 0.3)
        bf, bc = set(perm[:n1]), set(perm[n1:n1 + n2])
        fit = np.array([b in bf for b in books]); cal = np.array([b in bc for b in books]); draws.append((fit, cal, ~(fit | cal)))
    res = Parallel(n_jobs=16)(delayed(object_draw)(f, c, t, R[rows], Y, Erow, books, old) for f, c, t in draws)
    names = list(res[0])
    return {"pictures": int(len(rows)),
            "per_system": {n: {k: rb.summarise([r[n][k] for r in res]) for k in res[0][n]} for n in names},
            "system_minus": {n: {"mean_auroc": rb.paired(np.array([r["system"]["mean_auroc"] - r[n]["mean_auroc"] for r in res]))}
                             for n in names if n != "system"}}


# ---------------------------------------------------------------- scene presence
def presence_draw(tr, cal, te, e2e, V, rows, y, books, Erow):
    P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")]) @ rb.W
    auc = lambda yy, m, X: roc_auc_score(yy, m.predict_proba(X)[:, 1])
    Pp, C = probe(Erow[tr], y[tr], books[tr], Erow, False, auc)
    out = {}
    for name, s in (("system", P), ("linear probe", Pp)):
        err = ((s >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(s - 0.5)
        o = {"scene_auroc": float(roc_auc_score(y[te], s[te]))}
        for al in (0.05, 0.10):
            lam = rb.lam_for(u[cal], err[cal], al); auto = u[te] < lam
            o[f"approved@{al}"] = float(auto.mean()); o[f"risk@{al}"] = float(np.mean(err[te] * auto))
        out[name] = o
    out["linear probe"]["C"] = C
    return out


def presence_part(e2e, idx, nat, scene_abs, E):
    V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values
    draws = list(redraws(books))
    res = Parallel(n_jobs=16)(delayed(presence_draw)(tr, cal, te, e2e, V, rows, y, books, E[rows]) for tr, cal, te in draws)
    return {"pictures": int(len(sp)),
            "per_system": {n: {k: rb.summarise([r[n][k] for r in res]) for k in res[0][n] if k != "C"} for n in res[0]},
            "system_minus": {"linear probe": {k: rb.paired(np.array([r["system"][k] - r["linear probe"][k] for r in res]))
                                              for k in ("scene_auroc", "approved@0.05", "approved@0.1")}}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    E = embeddings(idx); E_by_id = {pid: E[r] for pid, r in idx.items()}
    out = {"plan": "self_annotation_package/more_experiments/PLAN_baselines.md (commit 08e2412)",
           "scene_classes": scene_part(idx, V, E), "objects": object_part(E_by_id),
           "scene_presence": presence_part(e2e, idx, nat, scene_abs, E)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for t in ("scene_classes", "objects", "scene_presence"):
        print("==", t)
        for n, z in out[t]["per_system"].items(): print(f"  {n:24s}", {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in z.items()})
        for n, z in out[t]["system_minus"].items(): print(f"  system - {n:22s}", {k: (v["mean"], v["p2.5"], v["p97.5"], v["share_of_draws_above_zero"]) for k, v in z.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
