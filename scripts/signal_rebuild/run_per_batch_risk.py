#!/usr/bin/env python3
"""How often does a single batch break the risk promise, and what does a stricter level cost?

Conformal risk control bounds the expected share of pictures approved and wrong, over draws of the
calibration set; a single batch can exceed the level. For the corrected design as reported (agents with
the "no scene" score, fixed line 0.5, uncertainty 1 - |p - 0.5|), over the same 200 book-disjoint
redraws as best_config_repeated.json, this records each redraw's realised risk on its test books for a
grid of calibration levels, and reports per level: mean approval, mean realised risk, and the share of
redraws whose realised risk exceeds 0.05 (a promise of 5 wrong in 100) and exceeds the level itself.
The level at which at most one batch in ten exceeds 0.05 is what an archive needing a per-batch
promise would use.

    python run_per_batch_risk.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_scene_definition_check as sd

OUT = rb.SR / "per_batch_risk.json"
LEVELS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
TARGET = 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    risk = {l: [] for l in LEVELS}; appr = {l: [] for l in LEVELS}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
        p = P @ rb.W; err = ((p >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(p - 0.5)
        for l in LEVELS:
            auto = sd.approve(u[cal], err[cal], u[te], l)
            risk[l].append(float(np.mean(err[te] * auto))); appr[l].append(float(auto.mean()))
    res = {str(l): {"mean_approved": round(float(np.mean(appr[l])), 4), "mean_realised_risk": round(float(np.mean(risk[l])), 4),
                    "share_over_0.05": round(float(np.mean(np.array(risk[l]) > TARGET)), 4),
                    "share_over_level": round(float(np.mean(np.array(risk[l]) > l)), 4)} for l in LEVELS}
    ok = [l for l in LEVELS if res[str(l)]["share_over_0.05"] <= 0.10]
    out = {"experiment": "Per-batch risk of the corrected design (with the no-scene score) at stricter levels",
           "pictures": int(len(y)), "draws": len(risk[LEVELS[0]]), "seed": rb.SEED, "target": TARGET,
           "levels": res, "loosest_level_with_at_most_10pct_batches_over_0.05": max(ok) if ok else None}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for l in LEVELS:
        r = res[str(l)]
        print(f"level {l:.3f}: approved {r['mean_approved']:.3f}  risk {r['mean_realised_risk']:.4f}  over 0.05 in {r['share_over_0.05']:.3f}  over level in {r['share_over_level']:.3f}")
    print("loosest level with <=10% of batches over 0.05:", out["loosest_level_with_at_most_10pct_batches_over_0.05"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
