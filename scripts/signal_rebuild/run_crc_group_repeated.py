#!/usr/bin/env python3
"""E3: does the risk guarantee hold per group, over 500 splits rather than one?

On one split of the current pool, the marginal bound at alpha = 0.05 was met while the group the
pipeline calls "scene" ran above target; one split of the expanded pool suggested the per-group
version held at a single level only. One split is one draw, so both are repeated here over the same
500 publication-level splits used for the marginal check.

Groups are the pipeline's own decision (predicted scene / predicted no scene), known at deployment.
Risk within a group is the share of that group's pictures that are approved errors.
  marginal    one threshold calibrated on all calibration pictures (as deployed)
  per group   a threshold calibrated within each group, finite-sample correction per group (Mondrian)
Score, error label and split rule are those of the deployed conformal risk control.

    python run_crc_group_repeated.py [--verify]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/signal_rebuild"))
from run_crc_repeated_splits import assign, POOLS, AGENTS, GRID, COLS, ALPHAS, N_SPLITS

OUT = BASE / "results/multi_agent/expanded/crc_group_repeated.json"
GROUPS = {"predicted_scene": 1, "predicted_no_scene": 0}


def lam_for(U, E, alpha):
    n = len(U)
    if n == 0: return GRID[0]
    risk = (E[None, :] * (U[None, :] < GRID[:, None])).mean(axis=1)
    ok = (n / (n + 1)) * risk + 1 / (n + 1) <= alpha
    return GRID[ok].max() if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    agents = pd.read_csv(AGENTS).drop_duplicates("image_id")
    out = {"experiment": "Per-group risk control over 500 publication-level splits (E3)", "splits": N_SPLITS, "pools": {}}
    for pool, path in POOLS.items():
        df = pd.read_csv(path).drop_duplicates("ppn_id").merge(agents, left_on="ppn_id", right_on="image_id", how="left")
        U = (0.6 * df[COLS].std(axis=1) + 0.4 * (1 - df[COLS].mean(axis=1))).values
        pred = (df["comparison_fusion_score"].values >= 0.5).astype(int)
        E = (pred != df["has_scene"].values).astype(int)
        rec = {m: {al: {"overall": [], "approval": [], **{g: [] for g in GROUPS}} for al in ALPHAS} for m in ("marginal", "per_group")}
        for seed in range(N_SPLITS):
            sp = assign(df, seed); c, t = sp == "calibration", sp == "test"
            for al in ALPHAS:
                lam_m = lam_for(U[c], E[c], al)
                lam_g = {g: lam_for(U[c & (pred == v)], E[c & (pred == v)], al) for g, v in GROUPS.items()}
                for m in ("marginal", "per_group"):
                    th = np.where(pred == 1, lam_g["predicted_scene"], lam_g["predicted_no_scene"]) if m == "per_group" else np.full(len(U), lam_m)
                    auto = (U < th) & t
                    r = rec[m][al]
                    r["overall"].append(float((E * auto)[t].mean())); r["approval"].append(float(auto[t].mean()))
                    for g, v in GROUPS.items():
                        mask = t & (pred == v)
                        r[g].append(float((E * auto)[mask].mean()) if mask.any() else 0.0)
        summ = {}
        for m, byal in rec.items():
            summ[m] = [{"alpha": al,
                        "mean_overall_risk": round(float(np.mean(v["overall"])), 4),
                        "mean_auto_approval_pct": round(100 * float(np.mean(v["approval"])), 1),
                        **{f"{g}_mean_risk": round(float(np.mean(v[g])), 4) for g in GROUPS},
                        **{f"{g}_mean_within_alpha": bool(np.mean(v[g]) <= al) for g in GROUPS},
                        **{f"{g}_share_splits_over_alpha": round(float(np.mean(np.array(v[g]) > al)), 3) for g in GROUPS}}
                       for al, v in byal.items()]
        out["pools"][pool] = summ
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for pool, summ in out["pools"].items():
        print(f"\n{pool}")
        for m, rows in summ.items():
            for r in rows:
                print(f"  {m:9s} a={r['alpha']:.2f}  overall {r['mean_overall_risk']:.4f}  approval {r['mean_auto_approval_pct']}%  "
                      f"scene group {r['predicted_scene_mean_risk']:.4f} ({'ok' if r['predicted_scene_mean_within_alpha'] else 'OVER'})  "
                      f"no-scene group {r['predicted_no_scene_mean_risk']:.4f} ({'ok' if r['predicted_no_scene_mean_within_alpha'] else 'OVER'})")


if __name__ == "__main__":
    raise SystemExit(main())
