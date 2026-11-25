#!/usr/bin/env python3
"""Platt scaling of the deployed fused score on C1, against the human label.

Section 4.5 calibrates the deployed fused score by Platt scaling on half of the 801 expert-reviewed
pictures (C1) and measures the calibration error on the other half. The figure earlier quoted there
(ECE 0.0007) came from a different computation: all 12,110 corpus pictures, calibrated against
"the scene agent's own score is above 0.5", one of the system's own signals rather than a human
label. This measures what the section describes, joined on the full picture name (book and page):
the fused score (comparison_fusion_score), the expert scene label, a random half for fitting and the
other half for testing, over 200 random halvings; ECE with 15 equal-width bins. Because 796 of the
801 pictures are scenes, it also reports the ECE of always predicting the fitting half's base rate.

    python run_c1_platt_check.py [--verify]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/core_pipeline"))
from picture_ids import full_id, read_expert_annotations

OUT = BASE / "results/signal_rebuild/c1_platt_check.json"
EXPERT = BASE / "human_spatial_audit/user_annotations_800.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
EXPERT_ROWS, DRAWS, SEED, BINS = 801, 200, 20261103, 15


def ece(p, y, bins=BINS):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def summarise(x):
    x = np.asarray(x)
    return {"mean": round(float(x.mean()), 4), "p2.5": round(float(np.percentile(x, 2.5)), 4), "p97.5": round(float(np.percentile(x, 97.5)), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    ex = read_expert_annotations(EXPERT).iloc[:EXPERT_ROWS].copy()
    ex["y"] = (ex.n_scene_labels > 0).astype(int)
    ag = pd.read_csv(AGENTS); ag["image_id"] = ag.image_id.map(full_id); ag = ag.drop_duplicates("image_id")
    j = ex.merge(ag[["image_id", "comparison_fusion_score"]], on="image_id", how="inner")
    s, y = j.comparison_fusion_score.values.astype(float), j.y.values
    rng = np.random.default_rng(SEED); res = {"raw": [], "platt": [], "base_rate": []}
    for _ in range(DRAWS):
        perm = rng.permutation(len(y)); fit, test = perm[: len(y) // 2], perm[len(y) // 2:]
        if len(set(y[fit])) < 2:
            continue
        m = LogisticRegression(C=100.0).fit(s[fit].reshape(-1, 1), y[fit])
        res["raw"].append(ece(np.clip(s[test], 0, 1), y[test]))
        res["platt"].append(ece(m.predict_proba(s[test].reshape(-1, 1))[:, 1], y[test]))
        res["base_rate"].append(ece(np.full(len(test), y[fit].mean()), y[test]))
    out = {"experiment": "Platt scaling of the deployed fused score on C1 against the expert label",
           "pictures": int(len(y)), "scenes": int(y.sum()), "draws_used": len(res["platt"]), "bins": BINS, "seed": SEED,
           "ece_test_half": {k: summarise(v) for k, v in res.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
