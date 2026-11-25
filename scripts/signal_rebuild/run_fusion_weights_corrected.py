#!/usr/bin/env python3
"""Why 0.6 / 0.2 / 0.2? The corrected pipeline's combination weights, tested against learned ones.

The corrected pipeline maps each agent's native vector to P(scene) by a fit on the train books, then
combines the three chances with the deployed Level-2 weights, 0.6 object, 0.2 scene, 0.2 VLM. Those
weights were hand-set for the deployed signals and never questioned for the corrected ones. This
compares, on the same 200 book-disjoint redraws as run_best_config_repeated.py (same seed, so the very
same splits), four ways of combining the three chances of corrected + absence:

  fixed     0.6 / 0.2 / 0.2, as reported
  equal     1/3 each
  learned   a logistic regression on the three chances, fitted on the train books only. The chances
            it is fitted on are out-of-fold (5 folds by book inside the train books), so it learns how
            far to trust each agent on pictures that agent's own mapping did not see
  object    the object finder's chance alone

Each draw scores scene AUROC and test error, and auto-approval with realised risk under confidence
triage at risk targets 0.05 and 0.10, calibrated on the calibration books. Paired differences against
the fixed weights carry the comparison.

    python run_fusion_weights_corrected.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import run_best_config_repeated as rb

BASE = Path("/data/brhanu/thesis_project")
OUT = BASE / "results/signal_rebuild/fusion_weights_corrected.json"
ALPHAS = [0.05, 0.10]
KEYS = ("object", "scene", "vlm")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e

    idx, nat, _old, scene_abs = e2e.native_vectors()
    V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT)
    sp = sp[sp.ppn_id.isin(idx)].copy()
    rows_all = sp.ppn_id.map(idx).values
    y = sp.has_scene.values
    books = sp.ppn.values
    uniq = np.unique(books)

    rules = ["fixed", "equal", "learned", "object"]
    acc = {r: {"scene_auroc": [], "test_error": [], **{f"conf@{al}": {"approved": [], "risk": []} for al in ALPHAS}}
           for r in rules}
    learned_w = []

    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books])
        te = ~(tr | cal)

        P = np.column_stack([e2e.to_probability(V[k], rows_all[tr], y[tr], rows_all) for k in KEYS])

        # out-of-fold chances on the train books, for fitting the combination
        oof = np.zeros((tr.sum(), 3))
        tr_rows, tr_y, tr_books = rows_all[tr], y[tr], books[tr]
        for f_tr, f_te in GroupKFold(n_splits=5).split(tr_rows, tr_y, tr_books):
            for j, k in enumerate(KEYS):
                oof[f_te, j] = e2e.to_probability(V[k], tr_rows[f_tr], tr_y[f_tr], tr_rows[f_te])
        stack = LogisticRegression(max_iter=2000, C=1.0).fit(oof, tr_y)
        c = stack.coef_[0]; learned_w.append(c / c.sum() if c.sum() > 0 else c)

        scores = {"fixed": P @ rb.W, "equal": P.mean(axis=1),
                  "learned": stack.predict_proba(P)[:, 1], "object": P[:, 0]}
        for r, s in scores.items():
            err = ((s >= 0.5).astype(int) != y).astype(int)
            acc[r]["scene_auroc"].append(float(roc_auc_score(y[te], s[te])))
            acc[r]["test_error"].append(float(err[te].mean()))
            u = 1 - np.abs(s - 0.5)
            for al in ALPHAS:
                lam = rb.lam_for(u[cal], err[cal], al)
                auto = u[te] < lam
                acc[r][f"conf@{al}"]["approved"].append(float(auto.mean()))
                acc[r][f"conf@{al}"]["risk"].append(float(np.mean(err[te] * auto)))

    lw = np.array(learned_w)
    out = {"experiment": "Corrected + absence: fixed 0.6/0.2/0.2 against equal, learned and object-only combination",
           "pictures": int(len(sp)), "books": int(len(uniq)), "draws": rb.N_DRAWS, "seed": rb.SEED,
           "split": "same 200 book-disjoint 60/20/20 redraws as best_config_repeated.json",
           "learned_weight_shares": {k: rb.summarise(lw[:, j]) for j, k in enumerate(KEYS)},
           "per_rule": {r: {"scene_auroc": rb.summarise(acc[r]["scene_auroc"]),
                            "test_error": rb.summarise(acc[r]["test_error"]),
                            **{f"confidence@{al}": {"auto_approved": rb.summarise(acc[r][f"conf@{al}"]["approved"]),
                                                    "mean_realised_risk": round(float(np.mean(acc[r][f"conf@{al}"]["risk"])), 4)}
                               for al in ALPHAS}}
                        for r in rules},
           "paired_vs_fixed": {}}
    for r in ("equal", "learned", "object"):
        out["paired_vs_fixed"][r] = {
            "scene_auroc": rb.paired(np.array(acc[r]["scene_auroc"]) - np.array(acc["fixed"]["scene_auroc"])),
            "test_error_reduction": rb.paired(np.array(acc["fixed"]["test_error"]) - np.array(acc[r]["test_error"])),
            **{f"auto_approved@{al}": rb.paired(np.array(acc[r][f"conf@{al}"]["approved"])
                                                 - np.array(acc["fixed"][f"conf@{al}"]["approved"])) for al in ALPHAS}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))

    f = lambda s: f"{s['mean']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"
    print(f"{rb.N_DRAWS} redraws, {len(uniq)} books, {len(sp)} pictures\n")
    print(f"{'':10s} {'scene AUROC':>24s} {'test error':>24s}  approved@0.05 (risk)  approved@0.10 (risk)")
    for r in rules:
        p = out["per_rule"][r]
        cells = "  ".join(f"{100*p[f'confidence@{al}']['auto_approved']['mean']:5.1f}% ({p[f'confidence@{al}']['mean_realised_risk']:.3f})"
                          for al in ALPHAS)
        print(f"  {r:8s} {f(p['scene_auroc']):>24s} {f(p['test_error']):>24s}  {cells}")
    print("\nlearned weight shares (object / scene / vlm):",
          " / ".join(f"{out['learned_weight_shares'][k]['mean']:.2f}" for k in KEYS))
    print("\npaired against fixed (mean [95% of draws], share of draws above zero)")
    for r, d in out["paired_vs_fixed"].items():
        for k, s in d.items():
            print(f"  {r:8s} {k:22s} {f(s)}  {100*s['share_of_draws_above_zero']:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
