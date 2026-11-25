#!/usr/bin/env python3
"""Conformal risk control over many publication-level splits, current vs expanded pool.

Conformal risk control guarantees that the test risk is at most alpha IN EXPECTATION
over the draw of calibration and test data. One split can exceed alpha while the
guarantee holds, and one split can satisfy it while the procedure is miscalibrated.
Reporting "bound met on the split" is therefore a single draw. This repeats the
exact split rule (stratified publication groups, 60/20/20) and the exact deployed
procedure (U = 0.6*sigma + 0.4*(1 - mean), finite-sample-corrected lambda) over
many seeds, for the current 1,250-picture pool and the pool expanded with the 285
contact-sheet labels.

    python run_crc_repeated_splits.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
POOLS = {"current_1250": BASE / "results/multi_agent/corrected_clean_pool.csv",
         "expanded_1535": BASE / "results/multi_agent/expanded/expanded_clean_pool.csv"}
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/multi_agent/expanded/crc_repeated_splits.json"
ALPHAS, N_SPLITS = [0.01, 0.02, 0.05, 0.10, 0.15], 500
GRID = np.linspace(0.001, 1.0, 1000)
COLS = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]


def assign(df, seed, train=0.60, cal=0.20):
    """The rule of build_corrected_clean_holdout_split.py, with the seed as a parameter."""
    g = df.groupby("ppn").agg(n=("ppn_id", "count"), pos_rate=("has_scene", "mean")).reset_index()
    g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    g["pos_bin"] = pd.qcut(g["pos_rate"], 5, labels=False, duplicates="drop")
    targets = {"train": train, "calibration": cal, "test": 1.0 - train - cal}
    running, out = {k: 0 for k in targets}, {}
    for _, b in g.groupby("pos_bin"):
        for _, r in b.sample(frac=1.0, random_state=seed).iterrows():
            tot = max(sum(running.values()), 1)
            k = max(targets, key=lambda s: targets[s] - running[s] / tot)
            out[r["ppn"]] = k; running[k] += r["n"]
    return df["ppn"].map(out).values


def run_pool(path, agents):
    df = pd.read_csv(path).drop_duplicates("ppn_id").merge(agents, left_on="ppn_id", right_on="image_id", how="left")
    assert df["comparison_fusion_score"].notna().all(), "a pool picture has no agent scores"
    U = (0.6 * df[COLS].std(axis=1) + 0.4 * (1 - df[COLS].mean(axis=1))).values
    E = ((df["comparison_fusion_score"].values >= 0.5).astype(int) != df["has_scene"].values).astype(int)
    per = {a: {"risk": [], "met": [], "approval": [], "recall": []} for a in ALPHAS}
    for seed in range(N_SPLITS):
        sp = assign(df, seed)
        c, t = sp == "calibration", sp == "test"
        n = c.sum()
        cal_risk = (E[c][None, :] * (U[c][None, :] < GRID[:, None])).mean(axis=1)
        for a in ALPHAS:
            ok = (n / (n + 1)) * cal_risk + 1 / (n + 1) <= a
            lam = GRID[ok].max() if ok.any() else 0.0
            auto = U[t] < lam
            risk = float((E[t] * auto).mean())
            per[a]["risk"].append(risk); per[a]["met"].append(risk <= a); per[a]["approval"].append(float(auto.mean()))
            per[a]["recall"].append(float(E[t][~auto].sum() / max(E[t].sum(), 1)))
    neg_test = int(((df["has_scene"] == 0).values & (assign(df, 42) == "test")).sum())
    return {"pictures": int(len(df)), "negatives": int((df["has_scene"] == 0).sum()), "test_negatives_seed42": neg_test,
            "per_alpha": [{"alpha": a,
                           "mean_test_risk": round(float(np.mean(v["risk"])), 4),
                           "mean_within_alpha": bool(np.mean(v["risk"]) <= a),
                           "share_of_splits_exceeding_alpha": round(1 - float(np.mean(v["met"])), 3),
                           "test_risk_p5_p95": [round(float(x), 4) for x in np.percentile(v["risk"], [5, 95])],
                           "mean_auto_approval_pct": round(100 * float(np.mean(v["approval"])), 1),
                           "mean_error_recall_pct": round(100 * float(np.mean(v["recall"])), 1)} for a, v in per.items()]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); args = ap.parse_args()
    agents = pd.read_csv(AGENTS).drop_duplicates("image_id")
    out = {"experiment": "Conformal risk control over repeated publication-level splits",
           "splits": N_SPLITS, "reading": "The guarantee is on mean_test_risk; share_of_splits_exceeding_alpha shows how often one split alone would look like a failure.",
           "pools": {k: run_pool(p, agents) for k, p in POOLS.items()}}
    if args.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in out["pools"].items():
        print(f"\n{k}: {v['pictures']} pictures, {v['negatives']} negatives")
        for r in v["per_alpha"]:
            print(f"  alpha {r['alpha']:.2f}  mean risk {r['mean_test_risk']:.4f} ({'within' if r['mean_within_alpha'] else 'ABOVE'})  "
                  f"single split exceeds {100*r['share_of_splits_exceeding_alpha']:.0f}%  p5-p95 {r['test_risk_p5_p95']}  "
                  f"approval {r['mean_auto_approval_pct']}%  recall {r['mean_error_recall_pct']}%")


if __name__ == "__main__":
    raise SystemExit(main())
