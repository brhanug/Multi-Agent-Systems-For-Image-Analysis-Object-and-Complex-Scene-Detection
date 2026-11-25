#!/usr/bin/env python3
"""E1-E4 of self_annotation_package/more_experiments/PLAN.md (commit 01682a7).

  e1  weighted calibration without archive labels (covariate-shift weights from unlabelled pictures)
  e2  Learn then Test: a promise per calibration set
  e3  prediction sets with a promise per scene class (Mondrian)
  e4  the describer (LLaVA) alone as the system

    python run_more_experiments.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_best_config_repeated as rb
import run_scene_definition_check as sd
import run_final_check as fc
import run_confirm_check as cc
import run_scene_classes as rs

OUT = rb.SR / "more_experiments.json"
GRID = rb.GRID


def setup():
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp0 = pd.read_csv(rb.SPLIT); sp0 = sp0[sp0.ppn_id.isin(idx)].copy()
    t1, person = fc.labels(); t1 = t1[t1.label != "unsure"].copy(); t1["y"] = (t1.label == "scene").astype(int)
    t1["ppn"] = t1.ppn_id.str.split("/").str[0]; t1 = t1.reset_index(drop=True)
    t2, _ = cc.labelled(cc.CC / "items.csv", cc.CC / "labels.csv")
    one = sp0.copy(); pl = person[person.label != "unsure"].set_index("ppn_id").label; hit = one.ppn_id.isin(pl.index)
    one.loc[hit, "has_scene"] = (one.loc[hit, "ppn_id"].map(pl) == "scene").astype(int).values
    return e2e, idx, V, sp0, one, t1, t2


def redraws(books, seed=rb.SEED):
    uniq = np.unique(books); rng = np.random.default_rng(seed)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books])
        yield tr, cal, ~(tr | cal)


def fuse(e2e, V, rows_fit, y_fit, rows, agents=("object", "scene", "vlm"), w=rb.W):
    P = np.column_stack([e2e.to_probability(V[k], rows_fit, y_fit, rows) for k in agents])
    return P @ (np.asarray(w) if len(agents) == 3 else np.ones(1))


def weighted_cut(u_cal, e_cal, w_cal, w_test, alpha):
    """Per test picture: largest cut-off with (sum_i w_i L_i + w_j) / (sum_i w_i + w_j) <= alpha."""
    A = np.array([np.sum(w_cal * e_cal * (u_cal < g)) for g in GRID]); W = w_cal.sum()
    lam = np.zeros(len(w_test))
    for j, wj in enumerate(w_test):
        ok = A <= alpha * (W + wj) - wj
        lam[j] = GRID[ok].max() if ok.any() else -1.0
    return lam


def e1(e2e, idx, V, one, t1, t2):
    test = pd.concat([t1[["ppn_id", "y"]], t2[["ppn_id", "y"]]]).reset_index(drop=True)
    used = set(one.ppn_id) | set(pd.read_csv(fc.FC / "items.csv").ppn_id) | set(pd.read_csv(cc.CC / "items.csv").ppn_id)
    X = np.column_stack([V["object"], V["scene"], V["vlm"]]); ids = np.array(list(idx.keys())); rows_all = np.array(list(idx.values()))
    unl = rows_all[~np.isin(ids, list(used))]
    pool_rows = one.ppn_id.map(idx).values
    sc = StandardScaler().fit(X[np.concatenate([pool_rows, unl])])
    dom = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[np.concatenate([pool_rows, unl])]),
                                                         np.r_[np.zeros(len(pool_rows)), np.ones(len(unl))])
    def wt(r):
        p = dom.predict_proba(sc.transform(X[r]))[:, 1]; return p / (1 - p) * len(pool_rows) / len(unl)
    w_pool, w_test = wt(pool_rows), wt(test.ppn_id.map(idx).values)
    rows = pool_rows; y = one.has_scene.values; xr = test.ppn_id.map(idx).values; xy = test.y.values
    acc = {f"{v}@{a}": {"risk": [], "approved": []} for v in ("unweighted", "weighted") for a in (0.05, 0.10)}
    ess = []
    for tr, cal, te in redraws(one.ppn.values):
        p = fuse(e2e, V, rows[tr], y[tr], np.concatenate([rows, xr])); pp, px = p[:len(rows)], p[len(rows):]
        e_cal = ((pp[cal] >= 0.5).astype(int) != y[cal]).astype(int); u_cal = 1 - np.abs(pp[cal] - 0.5)
        u_t = 1 - np.abs(px - 0.5); e_t = ((px >= 0.5).astype(int) != xy).astype(int)
        wc = w_pool[cal]; ess.append(float(wc.sum() ** 2 / (wc ** 2).sum()))
        for a in (0.05, 0.10):
            for v, wcal, wtest in (("unweighted", np.ones(cal.sum()), np.ones(len(xr))), ("weighted", wc, w_test)):
                lam = weighted_cut(u_cal, e_cal, wcal, wtest, a); ap = u_t < lam
                acc[f"{v}@{a}"]["risk"].append(float(np.mean(e_t * ap))); acc[f"{v}@{a}"]["approved"].append(float(ap.mean()))
    return {"archive_test_pictures": int(len(test)), "unlabelled_archive_pictures": int(len(unl)),
            "domain_classifier_auroc_in_sample": round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                np.r_[np.zeros(len(pool_rows)), np.ones(len(unl))], dom.predict_proba(sc.transform(X[np.concatenate([pool_rows, unl])]))[:, 1])), 4),
            "weight_on_archive_test_pictures": rb.summarise(w_test),
            "effective_calibration_size": rb.summarise(ess), "mean_calibration_size": round(float(np.mean([c.sum() for _, c, _ in redraws(one.ppn.values)])), 1),
            "results": {k: {"mean_realised_risk": round(float(np.mean(v["risk"])), 4), "share_of_redraws_over_alpha": round(float(np.mean(np.array(v["risk"]) > float(k.split("@")[1]))), 4),
                            "approved": rb.summarise(v["approved"])} for k, v in acc.items()}}


def ltt_cut(u, e, alpha, delta=0.10):
    n = len(u); last = None
    for g in GRID:
        k = int(np.sum(e * (u < g)))
        if binom.cdf(k, n, alpha) <= delta: last = g
        else: break
    return last


def e2(e2e, idx, V, sp0, one, t1, t2):
    out = {}
    rows = sp0.ppn_id.map(idx).values; y = sp0.has_scene.values
    acc = {f"{m}@{a}": {"risk": [], "approved": []} for m in ("crc", "ltt") for a in (0.05, 0.10)}
    for tr, cal, te in redraws(sp0.ppn.values):
        p = fuse(e2e, V, rows[tr], y[tr], rows); e = ((p >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(p - 0.5)
        for a in (0.05, 0.10):
            ap_c = sd.approve(u[cal], e[cal], u[te], a)
            g = ltt_cut(u[cal], e[cal], a); ap_l = (u[te] < g) if g is not None else np.zeros(te.sum(), bool)
            for m, ap in (("crc", ap_c), ("ltt", ap_l)):
                acc[f"{m}@{a}"]["risk"].append(float(np.mean(e[te] * ap))); acc[f"{m}@{a}"]["approved"].append(float(ap.mean()))
    out["pool_as_labelled"] = {k: {"mean_realised_risk": round(float(np.mean(v["risk"])), 4),
                                   "share_of_redraws_over_alpha": round(float(np.mean(np.array(v["risk"]) > float(k.split("@")[1]))), 4),
                                   "approved": rb.summarise(v["approved"])} for k, v in acc.items()}
    rows = one.ppn_id.map(idx).values; y = one.has_scene.values
    p1, p2 = fuse(e2e, V, rows, y, t1.ppn_id.map(idx).values), fuse(e2e, V, rows, y, t2.ppn_id.map(idx).values)
    e1_, e2_ = ((p1 >= 0.5).astype(int) != t1.y.values).astype(int), ((p2 >= 0.5).astype(int) != t2.y.values).astype(int)
    u1, u2 = 1 - np.abs(p1 - 0.5), 1 - np.abs(p2 - 0.5)
    arch = {}
    for a in (0.05, 0.10):
        ap_c = sd.approve(u1, e1_, u2, a); g = ltt_cut(u1, e1_, a); ap_l = (u2 < g) if g is not None else np.zeros(len(u2), bool)
        arch[f"alpha {a}"] = {m: {"realised_risk": round(float(np.mean(e2_ * ap)), 4), "approved": round(float(ap.mean()), 4)}
                              for m, ap in (("crc", ap_c), ("ltt", ap_l))}
    out["archive_cal_final_check_test_confirm"] = arch
    out["delta"] = 0.10
    return out


def e3(idx):
    import run_corrected_tier1_end_to_end as e2e
    _, nat, _o, scene_abs = e2e.native_vectors()
    ls = rs.llava_scene(idx)
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([ls, nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({a: i for i, a in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values; K = rs.K
    res = {}
    for variant in ("planned", "balanced"):
        rs.BALANCED = variant == "balanced"
        cov = {a: [] for a in rs.ANSWERS}; size, overall, marg_cov = [], [], {a: [] for a in rs.ANSWERS}; msize = []
        for tr, cal, te in redraws(d.ppn.values):
            Pk = {k: rs.to_probs(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")}
            P = 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"]
            q = np.full(K, -np.inf)
            for k in range(K):
                s = np.sort(P[cal][y[cal] == k, k]); m = int(np.floor(0.10 * (len(s) + 1)))
                if m >= 1: q[k] = s[m - 1]
            S = P[te] >= q[None, :]; S[np.arange(te.sum()), P[te].argmax(1)] = True
            g = rs.set_threshold(P[cal], y[cal], 0.10); S0 = rs.make_sets(P[te], g)
            yt = y[te]; c = S[np.arange(len(yt)), yt]; c0 = S0[np.arange(len(yt)), yt]
            for k in range(K):
                if (yt == k).any(): cov[rs.ANSWERS[k]].append(float(c[yt == k].mean())); marg_cov[rs.ANSWERS[k]].append(float(c0[yt == k].mean()))
            size.append(float(S.sum(1).mean())); overall.append(float(c.mean())); msize.append(float(S0.sum(1).mean()))
        res[variant] = {"per_class_sets": {"coverage_per_class": {a: round(float(np.mean(v)), 4) for a, v in cov.items()},
                                           "coverage": round(float(np.mean(overall)), 4), "set_size": round(float(np.mean(size)), 3)},
                        "one_threshold_sets": {"coverage_per_class": {a: round(float(np.mean(v)), 4) for a, v in marg_cov.items()},
                                               "set_size": round(float(np.mean(msize)), 3)}}
    rs.BALANCED = False
    return {"alpha": 0.10, "pictures": int(len(d)), "results": res}


def e4(e2e, idx, V, sp0, one, t1, t2):
    out = {}
    for name, ag in (("combined", ("object", "scene", "vlm")), ("describer alone", ("vlm",))):
        r = {}
        for lab, sp in (("as labelled", sp0), ("one rule", one)):
            rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; xr = t1.ppn_id.map(idx).values; xy = t1.y.values
            R, A, Rin, Ain = [], [], [], []
            for tr, cal, te in redraws(sp.ppn.values):
                p = fuse(e2e, V, rows[tr], y[tr], np.concatenate([rows, xr]), ag); pp, px = p[:len(rows)], p[len(rows):]
                e = ((pp >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(pp - 0.5)
                ap = sd.approve(u[cal], e[cal], 1 - np.abs(px - 0.5), 0.05); et = ((px >= 0.5).astype(int) != xy).astype(int)
                R.append(float(np.mean(et * ap))); A.append(float(ap.mean()))
                api = sd.approve(u[cal], e[cal], u[te], 0.05); Rin.append(float(np.mean(e[te] * api))); Ain.append(float(api.mean()))
            r[f"pool {lab}"] = {"in_pool": {"risk": round(float(np.mean(Rin)), 4), "approved": round(float(np.mean(Ain)), 4)},
                                "final_check_399": {"risk": round(float(np.mean(R)), 4), "approved": round(float(np.mean(A)), 4)}}
        rows = one.ppn_id.map(idx).values; y = one.has_scene.values
        p1, p2 = fuse(e2e, V, rows, y, t1.ppn_id.map(idx).values, ag), fuse(e2e, V, rows, y, t2.ppn_id.map(idx).values, ag)
        e1_, e2_ = ((p1 >= 0.5).astype(int) != t1.y.values).astype(int), ((p2 >= 0.5).astype(int) != t2.y.values).astype(int)
        ap = sd.approve(1 - np.abs(p1 - 0.5), e1_, 1 - np.abs(p2 - 0.5), 0.05)
        books = t1.ppn.values; ub = np.unique(books); grp = {b: np.where(books == b)[0] for b in ub}
        rng = np.random.default_rng(cc.SEED); RR, AA = [], []
        for _ in range(cc.BOOT):
            pick = np.concatenate([grp[b] for b in rng.choice(ub, len(ub), replace=True)])
            a_ = sd.approve(1 - np.abs(p1[pick] - 0.5), e1_[pick], 1 - np.abs(p2 - 0.5), 0.05); RR.append(float(np.mean(e2_ * a_))); AA.append(float(a_.mean()))
        r["archive_calibrated_confirm_294"] = {"as_drawn": {"risk": round(float(np.mean(e2_ * ap)), 4), "approved": round(float(ap.mean()), 4)},
                                                "over_calibration_resamples": {"risk": round(float(np.mean(RR)), 4), "approved": round(float(np.mean(AA)), 4)}}
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    e2e, idx, V, sp0, one, t1, t2 = setup()
    out = {"plan": "self_annotation_package/more_experiments/PLAN.md (commit 01682a7)",
           "E1_weighted_calibration": e1(e2e, idx, V, one, t1, t2),
           "E2_learn_then_test": e2(e2e, idx, V, sp0, one, t1, t2),
           "E3_per_class_sets": e3(idx),
           "E4_describer_alone": e4(e2e, idx, V, sp0, one, t1, t2)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
