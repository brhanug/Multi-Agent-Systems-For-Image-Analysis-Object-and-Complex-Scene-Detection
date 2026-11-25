#!/usr/bin/env python3
"""Risk control on each answer separately: can the corrected design approve "no scene" calls too?

With the decision line fitted on the training books (run_decision_threshold_check.py), the corrected
design makes "no scene" calls, but one risk promise over all pictures says nothing about how often an
approved "no scene" call is wrong: the scene calls, which are the majority, can carry the budget. Here
conformal risk control is run separately for each answer (pictures called scene, pictures called no
scene), each with its own promise at the same alpha, so that among the pictures given each answer the
share approved and wrong is bounded. Over the same 200 book-disjoint redraws, on the test books:

  joint       one promise over all pictures (as in decision_threshold_check.json)
  per answer  one promise per answer, calibrated on the calibration pictures given that answer

for the fitted line and, for reference, the fixed 0.5 line. Uncertainty is u = 1 - |p - line|.

    python run_two_sided_risk_control.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd
from run_decision_threshold_check import fitted_line

OUT = rb.SR / "two_sided_risk_control.json"
ALPHAS = [0.05, 0.10]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)

    keys = ["approved", "risk", "approved_scene_calls", "approved_no_scene_calls",
            "risk_within_scene_calls", "risk_within_no_scene_calls", "wrong_among_approved_no_scene"]
    acc = {f"{line} / {mode}": {f"{k}@{al}": [] for k in keys for al in ALPHAS}
           for line in ("fitted", "fixed 0.5") for mode in ("joint", "per answer")}
    groups = {l: {"cal_no_scene_calls": [], "test_no_scene_calls": [], "error_among_no_scene_calls": [],
                  "error_among_scene_calls": [], "auroc_within_no_scene_calls": [], "auroc_within_scene_calls": []}
              for l in ("fitted", "fixed 0.5")}
    pooled = {v: {al: [0, 0] for al in ALPHAS} for v in acc}   # wrong / approved no-scene calls over all redraws
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
        p = P @ rb.W
        for line, t in (("fitted", fitted_line(p[tr], y[tr])), ("fixed 0.5", 0.5)):
            c = (p >= t).astype(int); err = (c != y).astype(int); u = 1 - np.abs(p - t)
            groups[line]["cal_no_scene_calls"].append(int((c[cal] == 0).sum()))
            groups[line]["test_no_scene_calls"].append(int((c[te] == 0).sum()))
            for k, name in ((0, "no_scene_calls"), (1, "scene_calls")):
                g = te & (c == k)
                if g.any():
                    groups[line][f"error_among_{name}"].append(float(err[g].mean()))
                if len(set(err[g])) == 2:                 # does uncertainty rank the errors within one answer?
                    groups[line][f"auroc_within_{name}"].append(float(roc_auc_score(err[g], u[g])))
            for al in ALPHAS:
                joint = sd.approve(u[cal], err[cal], u[te], al)
                per = np.zeros(te.sum(), bool)
                for k in (0, 1):
                    gc, gt = cal & (c == k), c[te] == k
                    if gc.sum() and gt.sum():
                        per[gt] = sd.approve(u[gc], err[gc], u[te][gt], al)
                for mode, auto in (("joint", joint), ("per answer", per)):
                    d, e, ct = acc[f"{line} / {mode}"], err[te], c[te]
                    d[f"approved@{al}"].append(float(auto.mean())); d[f"risk@{al}"].append(float(np.mean(e * auto)))
                    for k, name in ((1, "scene_calls"), (0, "no_scene_calls")):
                        g = ct == k
                        if g.any():
                            d[f"approved_{name}@{al}"].append(float(auto[g].mean()))
                            d[f"risk_within_{name}@{al}"].append(float(np.mean(e[g] * auto[g])))
                    a0 = auto & (ct == 0)
                    if a0.any():
                        d[f"wrong_among_approved_no_scene@{al}"].append(float(e[a0].mean()))
                    pooled[f"{line} / {mode}"][al][0] += int(e[a0].sum()); pooled[f"{line} / {mode}"][al][1] += int(a0.sum())

    def s(k, x):
        return round(float(np.mean(x)), 4) if k.startswith("risk") else rb.summarise(x)
    out = {"experiment": "Conformal risk control per answer (scene / no scene) against one joint promise",
           "pictures": int(len(sp)), "draws": rb.N_DRAWS, "seed": rb.SEED,
           "uncertainty": "1 - |p - line|", "line_fit": "maximum balanced accuracy on the training books",
           "calls": {l: {k: rb.summarise(v) for k, v in g.items() if v} for l, g in groups.items()},
           "results": {v: {k: s(k, x) for k, x in d.items() if x} for v, d in acc.items()},
           "wrong_among_approved_no_scene_pooled": {v: {al: round(n0 / max(1, n), 4) for al, (n0, n) in d.items()}
                                                    for v, d in pooled.items()},
           "draws_with_an_approved_no_scene_call": {v: {al: len(d[f"wrong_among_approved_no_scene@{al}"]) for al in ALPHAS}
                                                    for v, d in acc.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["calls"], indent=1))
    for v, r in out["results"].items():
        print(f"\n{v}")
        for al in ALPHAS:
            g = lambda k: r.get(f"{k}@{al}")
            m = lambda k: g(k)["mean"] if isinstance(g(k), dict) else g(k)
            print(f"  alpha {al}: approved {m('approved')}  risk {m('risk')}  | scene calls approved {m('approved_scene_calls')} "
                  f"(risk {m('risk_within_scene_calls')})  no-scene calls approved {m('approved_no_scene_calls')} "
                  f"(risk {m('risk_within_no_scene_calls')}, wrong among approved {m('wrong_among_approved_no_scene')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
