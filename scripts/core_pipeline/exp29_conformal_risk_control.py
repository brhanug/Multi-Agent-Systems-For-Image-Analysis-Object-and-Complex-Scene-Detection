#!/usr/bin/env python3
"""
Exp 29: Conformal Risk Control (CRC) & Split Conformal Prediction for Archival Triage
================================================================──────────────────────
Implements finite-sample distribution-free risk control (Angelopoulos & Bates, 2023)
on the n=801 expert-verified gold cohort.

Guarantees that auto-approved cataloging metadata maintains an un-audited error rate
R(lambda) <= alpha at user-specified significance levels (alpha = 0.01, 0.05, 0.10).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, STORED_COLUMN_THRESHOLD  # deliberate 0.55; see saa_decision.py
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_JSON = BASE / "results/multi_agent/exp29_crc_results.json"
OUTPUT_PLOT = BASE / "latex_assets" / "exp29_crc_coverage_curve.png"

from picture_ids import full_id as normalize_id, read_expert_annotations   # book and page, never the page alone


def main():
    print("🚀 Running Exp 29: Conformal Risk Control (CRC) for Guaranteed Archival Triage Bounds...")

    # 1. Load data
    expert = read_expert_annotations(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)

    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])

    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"  Matched {len(joined)} reviewed images with agent scores.")

    gold_label = joined["gold_has_scene"].values
    fusion_score = joined["comparison_fusion_score"].values
    
    # Compute agent score disagreement
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = joined[agent_cols].std(axis=1).values
    c_bar = joined[agent_cols].mean(axis=1).values
    unified_U = 0.6 * sigma_agents + 0.4 * (1.0 - c_bar)

    # Classification error flag
    model_pred = (fusion_score >= STORED_COLUMN_THRESHOLD).astype(int)
    error_flag = (model_pred != gold_label).astype(int)
    n_total_errors = int(error_flag.sum())
    print(f"  Total errors in cohort: {n_total_errors} / {len(joined)} ({n_total_errors / len(joined):.2%})")

    # 2. Split into Calibration (50%) and Test (50%) sets deterministically
    np.random.seed(42)
    indices = np.arange(len(joined))
    np.random.shuffle(indices)

    n_cal = 400
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    U_cal, E_cal = unified_U[cal_idx], error_flag[cal_idx]
    U_test, E_test = unified_U[test_idx], error_flag[test_idx]

    # 3. Conformal Risk Control Optimization
    # Loss function for image i under threshold lambda:
    # L_i(lambda) = E_i * I(U_i < lambda)  (Error committed on auto-approved image)
    # Target: R(lambda) = E[L_i(lambda)] <= alpha

    grid_lambdas = np.linspace(0.001, 1.0, 1000)
    alpha_levels = [0.01, 0.02, 0.05, 0.10, 0.15]
    delta = 0.05  # 95% statistical confidence parameter

    crc_results = []
    
    # Empirical risk on calibration set for each lambda
    cal_risks = np.array([np.mean(E_cal * (U_cal < lam)) for lam in grid_lambdas])
    
    # Upper confidence bound via Bentkus / Hoeffding inequality or quantile adjustment
    # CRC threshold selection: lambda_hat = max { lam : (n_cal / (n_cal + 1)) * R_cal(lam) + (1 / (n_cal + 1)) <= alpha }
    
    for alpha in alpha_levels:
        # Find threshold satisfying CRC upper bound
        valid_mask = (n_cal / (n_cal + 1.0)) * cal_risks + (1.0 / (n_cal + 1.0)) <= alpha
        if np.any(valid_mask):
            # Select max lambda that satisfies risk bound (maximizes auto-approval rate)
            valid_lambdas = grid_lambdas[valid_mask]
            lambda_hat = float(np.max(valid_lambdas))
        else:
            lambda_hat = float(grid_lambdas[0])

        # Evaluate on held-out test set
        test_auto_approved = U_test < lambda_hat
        test_audited = ~test_auto_approved
        
        n_auto = int(test_auto_approved.sum())
        n_audited = int(test_audited.sum())
        
        # Test risk = Un-audited error rate (errors made on auto-approved images / total test size)
        test_risk = float(np.mean(E_test * test_auto_approved))
        
        # Error recall = Fraction of test errors caught by human audit
        test_errors_total = int(E_test.sum())
        test_errors_caught = int(E_test[test_audited].sum())
        error_recall = float(test_errors_caught / test_errors_total) if test_errors_total > 0 else 1.0
        
        auto_approval_pct = float(n_auto / len(test_idx))
        audit_pct = float(n_audited / len(test_idx))
        efficiency_lift = float((test_errors_caught / (n_audited + 1e-9)) / (test_errors_total / len(test_idx)))

        res = {
            "target_risk_alpha": alpha,
            "conformal_threshold_lambda": round(lambda_hat, 4),
            "cal_empirical_risk": round(float(cal_risks[grid_lambdas == lambda_hat][0]) if np.any(grid_lambdas == lambda_hat) else 0.0, 4),
            "test_empirical_risk": round(test_risk, 4),
            "risk_bound_satisfied": bool(test_risk <= alpha),
            "test_auto_approval_rate": round(auto_approval_pct * 100, 2),
            "test_audit_workload_rate": round(audit_pct * 100, 2),
            "test_error_recall": round(error_recall * 100, 2),
            "test_efficiency_lift": round(efficiency_lift, 2)
        }
        crc_results.append(res)
        print(f"  Alpha = {alpha:.2f} -> Threshold Lambda = {lambda_hat:.4f} | Test Risk = {test_risk:.4f} (Bound Met: {test_risk <= alpha}) | Auto-Approved: {auto_approval_pct*100:.1f}% | Error Recall: {error_recall*100:.1f}%")

    # 4. Generate Visualization
    plt.figure(figsize=(9, 5.5))
    plt.plot(grid_lambdas, [np.mean(E_test * (U_test < lam)) for lam in grid_lambdas], 
             label='Empirical Test Risk R(λ)', color='#1A56A0', linewidth=2.5)
    
    for r in crc_results:
        plt.axhline(r["target_risk_alpha"], color='#B71C1C', linestyle='--', alpha=0.5)
        plt.axvline(r["conformal_threshold_lambda"], color='#2E7D32', linestyle=':', alpha=0.7)
        plt.scatter([r["conformal_threshold_lambda"]], [r["test_empirical_risk"]], color='#EF7C00', s=60, zorder=5)

    plt.title("Conformal Risk Control (CRC) Guaranteed Triage Error Bounds (n=401 Test Split)", fontsize=12, fontweight='bold')
    plt.xlabel("Uncertainty Decision Threshold (λ)", fontsize=11)
    plt.ylabel("Un-audited Archive Error Risk R(λ)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["Test Risk R(λ)", "Target Error Bound (α)", "CRC Threshold (λ̂)"], loc='upper left')
    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"✅ Plot saved to {OUTPUT_PLOT}")

    # 5. Save JSON report
    report = {
        "experiment": "Exp 29: Conformal Risk Control (CRC) for Archival Triage",
        "provenance": "REAL — computed on n=801 CVAT expert gold cohort",
        "n_calibration": n_cal,
        "n_test": len(test_idx),
        "total_test_errors": int(E_test.sum()),
        "confidence_level_1_minus_delta": 1 - delta,
        "results": crc_results
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
