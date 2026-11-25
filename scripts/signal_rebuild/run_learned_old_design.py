#!/usr/bin/env python3
"""self_annotation_package/more_experiments/PLAN_learned_old.md (commit c68be99): the old design with its SAA
weights, SAA threshold and triage score all learned on the training books, against the deployed old design
and the corrected design, on the same 200 redraws.

    python run_learned_old_design.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import run_best_config_repeated as rb
from run_more_experiments import redraws

OUT = rb.SR / "learned_old_design.json"
ALPHAS = [0.05, 0.10]
WS = [round(0.1 * k, 1) for k in range(11)]
THREE = ["existing_pipeline_agent", "scene_agent", "vlm_agent"]


def route(u, err, y, s, cal, te):
    o = {"scene_auroc": float(roc_auc_score(y[te], s[te]))}
    for al in ALPHAS:
        lam = rb.lam_for(u[cal], err[cal], al); auto = u[te] < lam
        o[f"approved@{al}"] = float(auto.mean()); o[f"risk@{al}"] = float(np.mean(err[te] * auto))
    return o


def err_features(S, p):
    return np.column_stack([S, S.std(axis=1, ddof=1), S.mean(axis=1), np.abs(p - 0.5)])


def one_draw(tr, cal, te, S4, S3, y, books, Pc):
    out = {}
    # deployed old design
    saa = S3 @ np.array([0.6, 0.2, 0.2]); err = ((saa >= 0.5).astype(int) != y).astype(int)
    u = 0.6 * S4.std(axis=1, ddof=1) + 0.4 * (1 - S4.mean(axis=1))
    out["old, as deployed"] = route(u, err, y, saa, cal, te)
    for name, X in (("learned SAA (3 signals)", S3), ("learned SAA (4 signals)", S4)):
        p = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
        err = ((p >= 0.5).astype(int) != y).astype(int)
        # (a) the 0.6/0.4 form with the weight learned
        U = {w: w * S4.std(axis=1, ddof=1) + (1 - w) * (1 - S4.mean(axis=1)) for w in WS}
        trs = np.where(tr)[0]
        wb = max(WS, key=lambda w: roc_auc_score(err[trs], U[w][trs]) if 0 < err[trs].sum() < len(trs) else 0)
        out[f"{name}, triage weight learned"] = route(U[wb], err, y, p, cal, te) | {"w": wb}
        # (b) a learned error model, fitted on out-of-fold errors inside the training books
        oof_err = np.zeros(len(trs), int); oof_p = np.zeros(len(trs))
        for a, b in GroupKFold(5).split(trs, y[trs], books[trs]):
            m = LogisticRegression(max_iter=2000, C=1.0).fit(X[trs[a]], y[trs[a]])
            oof_p[b] = m.predict_proba(X[trs[b]])[:, 1]; oof_err[b] = ((oof_p[b] >= 0.5).astype(int) != y[trs[b]]).astype(int)
        F = err_features(S4, p); Ftr = err_features(S4[trs], oof_p)
        if 0 < oof_err.sum() < len(oof_err):
            ue = LogisticRegression(max_iter=2000, C=1.0).fit(Ftr, oof_err).predict_proba(F)[:, 1]
        else:
            ue = 1 - np.abs(p - 0.5)
        out[f"{name}, learned error model"] = route(ue, err, y, p, cal, te)
    errc = ((Pc >= 0.5).astype(int) != y).astype(int)
    out["corrected design"] = route(1 - np.abs(Pc - 0.5), errc, y, Pc, cal, te)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    y = sp.has_scene.values; books = sp.ppn.values; rows = sp.ppn_id.map(idx).values
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    S4, S3 = ag[rb.A4].values.astype(float), ag[THREE].values.astype(float)
    draws = list(redraws(books))
    Pcs = [np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")]) @ rb.W
           for tr, _c, _t in draws]
    res = Parallel(n_jobs=16)(delayed(one_draw)(tr, cal, te, S4, S3, y, books, Pc) for (tr, cal, te), Pc in zip(draws, Pcs))
    names = list(res[0])
    out = {"plan": "self_annotation_package/more_experiments/PLAN_learned_old.md (commit c68be99)", "pictures": int(len(sp)),
           "per_variant": {n: {k: rb.summarise([r[n][k] for r in res]) for k in res[0][n] if k != "w"} for n in names},
           "weights_chosen": {n: {str(w): int(sum(r[n]["w"] == w for r in res)) for w in WS} for n in names if "w" in res[0][n]},
           "corrected_minus": {n: {k: rb.paired(np.array([r["corrected design"][k] - r[n][k] for r in res]))
                                   for k in ("scene_auroc", "approved@0.05", "approved@0.1")} for n in names if n != "corrected design"}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for n, z in out["per_variant"].items(): print(f"{n:45s}", {k: v["mean"] for k, v in z.items()})
    for n, z in out["corrected_minus"].items(): print(f"corrected - {n:40s}", {k: (v["mean"], v["p2.5"], v["p97.5"]) for k, v in z.items()})
    print(out["weights_chosen"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
