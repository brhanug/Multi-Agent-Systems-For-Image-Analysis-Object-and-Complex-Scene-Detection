#!/usr/bin/env python3
"""
run_clean_holdout_crc.py
--------------------------
Recomputes Exp 29's Conformal Risk Control (CRC) triage policy on the
group-level (publication/PPN) train/calibration/test split produced by
build_clean_holdout_split.py, instead of exp29_conformal_risk_control.py's
original random 50/50 image-level split of C1.

Why this exists
----------------
exp29_conformal_risk_control.py draws its "calibration" (n=400) and "test"
(n=401) sets by randomly shuffling the *same* n=801 C1 cohort and splitting
50/50 — i.e. calibration and test images can come from (and typically do
come from) the same publications, and C1 itself is the same cohort used
upstream for agent-weight selection and AWLF development. This is exactly
the leakage class this thesis's group-level re-evaluation (see
build_clean_holdout_split.py, run_clean_holdout_evaluation.py) was built to
rule out for the SAA/AWLF/SCI-routing numbers — but the CRC number, which
this thesis promotes to its primary headline certified-triage claim, was
never put through the same test. This script closes that gap.

Protocol
--------
  train        -> not used by CRC (no classifier being fit; kept unused for
                   symmetry with the other clean-holdout scripts)
  calibration  -> CRC threshold (lambda_hat) selection only, exactly as in
                   exp29 (Bentkus/Hoeffding-style upper-confidence-bound rule)
  test         -> ALL reported numbers come from this partition alone; its
                   publications never appear in calibration.

Output: results/multi_agent/corrected_clean_holdout_crc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SPLIT_CSV = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_JSON = BASE / "results/multi_agent/corrected_clean_holdout_crc.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against the committed JSON; write nothing")
    ap.add_argument("--split-csv", default=None, help="read another split, e.g. the expanded pool")
    ap.add_argument("--out-json", default=None, help="write or verify another output file")
    args = ap.parse_args()
    global SPLIT_CSV, OUT_JSON
    if args.split_csv: SPLIT_CSV = Path(args.split_csv)
    if args.out_json: OUT_JSON = Path(args.out_json)
    split = pd.read_csv(SPLIT_CSV)
    agents = pd.read_csv(AGENTS_CSV).drop_duplicates("image_id")
    joined = split.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    print(f"Matched {len(joined)}/{len(split)} split images with agent scores.")

    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = joined[agent_cols].std(axis=1).values
    c_bar = joined[agent_cols].mean(axis=1).values
    joined["unified_U"] = 0.6 * sigma_agents + 0.4 * (1.0 - c_bar)

    model_pred = (joined["comparison_fusion_score"].values >= 0.5).astype(int)
    gold = joined["has_scene"].values
    joined["error_flag"] = (model_pred != gold).astype(int)

    cal = joined[joined["split"] == "calibration"].copy()
    test = joined[joined["split"] == "test"].copy()
    print(f"  calibration={len(cal)}  test={len(test)}  "
          f"(test error rate={test['error_flag'].mean():.3f})")

    U_cal, E_cal = cal["unified_U"].values, cal["error_flag"].values
    U_test, E_test = test["unified_U"].values, test["error_flag"].values
    n_cal = len(U_cal)

    grid_lambdas = np.linspace(0.001, 1.0, 1000)
    alpha_levels = [0.01, 0.02, 0.05, 0.10, 0.15]

    cal_risks = np.array([np.mean(E_cal * (U_cal < lam)) for lam in grid_lambdas])

    crc_results = []
    for alpha in alpha_levels:
        valid_mask = (n_cal / (n_cal + 1.0)) * cal_risks + (1.0 / (n_cal + 1.0)) <= alpha
        if np.any(valid_mask):
            lambda_hat = float(np.max(grid_lambdas[valid_mask]))
        else:
            lambda_hat = float(grid_lambdas[0])

        test_auto_approved = U_test < lambda_hat
        test_audited = ~test_auto_approved

        n_auto = int(test_auto_approved.sum())
        n_audited = int(test_audited.sum())

        test_risk = float(np.mean(E_test * test_auto_approved))
        test_errors_total = int(E_test.sum())
        test_errors_caught = int(E_test[test_audited].sum())
        error_recall = float(test_errors_caught / test_errors_total) if test_errors_total > 0 else 1.0

        auto_approval_pct = float(n_auto / len(test))
        audit_pct = float(n_audited / len(test))
        efficiency_lift = float((test_errors_caught / (n_audited + 1e-9)) / (test_errors_total / len(test)))

        res = {
            "target_risk_alpha": alpha,
            "conformal_threshold_lambda": round(lambda_hat, 4),
            "test_empirical_risk": round(test_risk, 4),
            "risk_bound_satisfied": bool(test_risk <= alpha),
            "test_auto_approval_rate": round(auto_approval_pct * 100, 2),
            "test_audit_workload_rate": round(audit_pct * 100, 2),
            "test_error_recall": round(error_recall * 100, 2),
            "test_efficiency_lift": round(efficiency_lift, 2),
        }
        crc_results.append(res)
        print(f"  Alpha={alpha:.2f}  lambda={lambda_hat:.4f}  TestRisk={test_risk:.4f} "
              f"(bound met: {test_risk <= alpha})  AutoApprove={auto_approval_pct*100:.1f}%  "
              f"Recall={error_recall*100:.1f}%  Lift={efficiency_lift:.2f}x")

    report = {
        "experiment": "Clean group-level re-evaluation of Exp 29 (Conformal Risk Control)",
        "protocol": "Group-level (publication/PPN) split. Calibration used ONLY for lambda "
                     "selection; all reported numbers are on the test partition, whose "
                     "publications never appear in calibration.",
        "n_calibration": len(cal),
        "n_test": len(test),
        "test_error_rate": float(test["error_flag"].mean()),
        "results": crc_results,
    }
    if args.verify:
        prev = json.loads(OUT_JSON.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(
            json.loads(json.dumps(report)), sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
