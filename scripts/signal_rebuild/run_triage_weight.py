#!/usr/bin/env python3
"""self_annotation_package/more_experiments/PLAN_triage_weight.md (commit 50b7a89): the mixing weight of the
deployed triage score U_w = w * spread + (1 - w) * (1 - mean confidence), w = 0 .. 1, on the 200 redraws.

    python run_triage_weight.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb

OUT = rb.SR / "triage_weight.json"
WS = [round(0.1 * k, 1) for k in range(11)]
ALPHAS = [0.05, 0.10]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, _nat, _old, _s = e2e.native_vectors()
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    S = ag[rb.A4].values
    saa = 0.6 * ag.existing_pipeline_agent.values + 0.2 * ag.scene_agent.values + 0.2 * ag.vlm_agent.values
    err = ((saa >= 0.5).astype(int) != y).astype(int)
    spread = S.std(axis=1, ddof=1); conf = 1 - S.mean(axis=1)
    U = {w: w * spread + (1 - w) * conf for w in WS}
    # extra, added after the planned run: the corrected design (with the "no scene" score) on the same redraws,
    # against the deployed triage score as deployed (w = 0.6)
    idx_, nat, _o2, scene_abs = e2e.native_vectors(); Vc = {**nat, "scene": scene_abs}; rows = sp.ppn_id.map(idx).values
    corr = {al: [] for al in ALPHAS}
    acc = {w: {"error_auroc": [], **{f"approved@{al}": [] for al in ALPHAS}, **{f"risk@{al}": [] for al in ALPHAS}} for w in WS}
    learned = {"w": [], "error_auroc": [], **{f"approved@{al}": [] for al in ALPHAS}}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        res = {}
        for w in WS:
            u = U[w]; z = acc[w]
            z["error_auroc"].append(float(roc_auc_score(err[te], u[te])))
            for al in ALPHAS:
                lam = rb.lam_for(u[cal], err[cal], al); auto = u[te] < lam
                z[f"approved@{al}"].append(float(auto.mean())); z[f"risk@{al}"].append(float(np.mean(err[te] * auto)))
        P = np.column_stack([e2e.to_probability(Vc[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")]) @ rb.W
        ec = ((P >= 0.5).astype(int) != y).astype(int); uc = 1 - np.abs(P - 0.5)
        for al in ALPHAS:
            lam = rb.lam_for(uc[cal], ec[cal], al); corr[al].append(float((uc[te] < lam).mean()))
        wb = max(WS, key=lambda w: roc_auc_score(err[tr], U[w][tr]))
        learned["w"].append(wb); learned["error_auroc"].append(acc[wb]["error_auroc"][-1])
        for al in ALPHAS: learned[f"approved@{al}"].append(acc[wb][f"approved@{al}"][-1])
    out = {"plan": "self_annotation_package/more_experiments/PLAN_triage_weight.md (commit 50b7a89)", "pictures": int(len(sp)),
           "per_w": {str(w): {m: rb.summarise(v) for m, v in z.items()} for w, z in acc.items()},
           "learned_w": {"chosen": {str(w): int(sum(x == w for x in learned["w"])) for w in WS},
                         **{m: rb.summarise(v) for m, v in learned.items() if m != "w"}},
           "extra_corrected_minus_deployed_triage": {f"approved@{al}": {"corrected": rb.summarise(corr[al]),
                                                                       "difference": rb.paired(np.array(corr[al]) - np.array(acc[0.6][f"approved@{al}"]))}
                                                      for al in ALPHAS},
           "w0_minus_w0.6": {m: rb.paired(np.array(acc[0.0][m]) - np.array(acc[0.6][m])) for m in ("error_auroc", "approved@0.05", "approved@0.1")}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for w, z in out["per_w"].items(): print(w, {m: v["mean"] for m, v in z.items()})
    print("learned", out["learned_w"]); print("w0 - w0.6", out["w0_minus_w0.6"]); return 0


if __name__ == "__main__":
    raise SystemExit(main())
