#!/usr/bin/env python3
"""Ablation of the corrected pipeline: what each agent and the absence score contribute.

Over the same 200 book-disjoint redraws as run_best_config_repeated.py (same seed, same splits),
the corrected + absence pipeline is scored with pieces removed:

  full              all three agents, scene matcher with the "no scene" score, 0.6 / 0.2 / 0.2
  no absence        scene matcher without the "no scene" score (5 scene scores only)
  no object finder  scene matcher + describer, 0.5 / 0.5
  no scene matcher  object finder + describer, 0.75 / 0.25 (the full weights, renormalised)
  no describer      object finder + scene matcher, 0.75 / 0.25
  object alone, scene alone, describer alone

Each draw refits every agent's mapping on its train books, calibrates the risk threshold on its
calibration books (confidence triage, risk target 0.05) and measures on its test books. Reported:
scene AUROC, test error, auto-approval and realised risk at 0.05, and for each variant the paired
difference from the full pipeline (mean, 95% of draws, share of draws in which removing the piece
made things worse).

    python run_ablation_corrected.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb

BASE = Path("/data/brhanu/thesis_project")
OUT = BASE / "results/signal_rebuild/ablation_corrected.json"
ALPHA = 0.05
VARIANTS = {
    "full":             ({"object": 0.6, "scene": 0.2, "vlm": 0.2}, True),
    "no absence":       ({"object": 0.6, "scene": 0.2, "vlm": 0.2}, False),
    "no object finder": ({"scene": 0.5, "vlm": 0.5}, True),
    "no scene matcher": ({"object": 0.75, "vlm": 0.25}, True),
    "no describer":     ({"object": 0.75, "scene": 0.25}, True),
    "object alone":     ({"object": 1.0}, True),
    "scene alone":      ({"scene": 1.0}, True),
    "describer alone":  ({"vlm": 1.0}, True),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e

    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(rb.SPLIT)
    sp = sp[sp.ppn_id.isin(idx)].copy()
    rows_all = sp.ppn_id.map(idx).values
    y = sp.has_scene.values
    books = sp.ppn.values
    uniq = np.unique(books)

    acc = {v: {"auroc": [], "error": [], "approved": [], "risk": []} for v in VARIANTS}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books])
        te = ~(tr | cal)

        P = {"object": e2e.to_probability(nat["object"], rows_all[tr], y[tr], rows_all),
             "vlm": e2e.to_probability(nat["vlm"], rows_all[tr], y[tr], rows_all),
             "scene_abs": e2e.to_probability(scene_abs, rows_all[tr], y[tr], rows_all),
             "scene_plain": e2e.to_probability(nat["scene"], rows_all[tr], y[tr], rows_all)}
        for v, (w, absence) in VARIANTS.items():
            s = sum(wt * (P["scene_abs" if absence else "scene_plain"] if k == "scene" else P[k])
                    for k, wt in w.items())
            err = ((s >= 0.5).astype(int) != y).astype(int)
            u = 1 - np.abs(s - 0.5)
            lam = rb.lam_for(u[cal], err[cal], ALPHA)
            auto = u[te] < lam
            acc[v]["auroc"].append(float(roc_auc_score(y[te], s[te])))
            acc[v]["error"].append(float(err[te].mean()))
            acc[v]["approved"].append(float(auto.mean()))
            acc[v]["risk"].append(float(np.mean(err[te] * auto)))

    full = {k: np.array(v) for k, v in acc["full"].items()}
    out = {"experiment": "Ablation of the corrected + absence pipeline over 200 book-disjoint redraws",
           "pictures": int(len(sp)), "books": int(len(uniq)), "draws": rb.N_DRAWS, "seed": rb.SEED,
           "risk_target": ALPHA, "triage": "confidence",
           "variants": {v: {"weights": w, "absence_score": ab} for v, (w, ab) in VARIANTS.items()},
           "per_variant": {v: {"scene_auroc": rb.summarise(acc[v]["auroc"]),
                               "test_error": rb.summarise(acc[v]["error"]),
                               "auto_approved": rb.summarise(acc[v]["approved"]),
                               "mean_realised_risk": round(float(np.mean(acc[v]["risk"])), 4)}
                           for v in VARIANTS},
           "loss_from_removing": {}}
    for v in VARIANTS:
        if v == "full":
            continue
        out["loss_from_removing"][v] = {
            "scene_auroc": rb.paired(full["auroc"] - np.array(acc[v]["auroc"])),
            "test_error": rb.paired(np.array(acc[v]["error"]) - full["error"]),
            "auto_approved": rb.paired(full["approved"] - np.array(acc[v]["approved"]))}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))

    f = lambda s: f"{s['mean']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"
    print(f"{rb.N_DRAWS} redraws, {len(uniq)} books, {len(sp)} pictures; triage by confidence at {ALPHA}\n")
    print(f"{'':18s} {'scene AUROC':>24s} {'test error':>24s} {'approved':>24s}  risk")
    for v in VARIANTS:
        p = out["per_variant"][v]
        print(f"  {v:16s} {f(p['scene_auroc']):>24s} {f(p['test_error']):>24s} {f(p['auto_approved']):>24s}  {p['mean_realised_risk']:.3f}")
    print("\nloss from removing (full minus variant; positive = the piece helps), share of draws worse without it")
    for v, d in out["loss_from_removing"].items():
        print(f"  {v:16s} " + "  ".join(f"{k}: {f(s)} {100*s['share_of_draws_above_zero']:.0f}%" for k, s in d.items()))


if __name__ == "__main__":
    raise SystemExit(main())
