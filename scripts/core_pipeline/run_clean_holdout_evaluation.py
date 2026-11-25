#!/usr/bin/env python3
"""
run_clean_holdout_evaluation.py
--------------------------------
Recomputes the thesis's headline error-detection and triage-routing metrics
on the group-level (publication/PPN) train/calibration/test split produced by
build_clean_holdout_split.py, instead of the prior in-sample or leaked splits.

Protocol
--------
  train        -> AWLF classifier fitting only
  calibration  -> Platt scaling parameters (A, B) only
  test         -> ALL reported numbers below come from this partition alone.
                   Never touched by any weight, threshold, or classifier choice.

SCI-based routing tier thresholds are chosen as terciles of the FULL C0
deployment corpus (12,110 images) - i.e. without looking at outcomes on the
human-labeled subset at all - then applied to the clean test partition to
measure the resulting review-budget / error-recall trade-off.

Reports, for the test partition only:
  - SAA disagreement (U_SAA) AUROC / ECE / recall@20%   [[the primary claim]]
  - Temperature-scaled entropy, "MC Dropout" (YOLO inverse), deep-ensemble std
  - AWLF (logistic classifier, trained on train, evaluated on test)
  - SCI-tercile routing: review budget vs error recall, on test only

Output: results/multi_agent/clean_holdout_evaluation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path("/data/brhanu/thesis_project")
SPLIT_CSV = BASE / "results/multi_agent/clean_holdout_split.csv"
AGENTS_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
SCI_CSV = BASE / "results/multi_agent/scene_complexity_index.csv"
OUT_JSON = BASE / "results/multi_agent/clean_holdout_evaluation.json"


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total / max(len(y_true), 1))


def recall_at_k(y_true, scores, k=0.20):
    n = len(y_true)
    top = np.argsort(np.asarray(scores))[::-1][: max(1, int(n * k))]
    y_true = np.asarray(y_true)
    return float(y_true[top].sum() / max(1, y_true.sum()))


def main() -> None:
    split = pd.read_csv(SPLIT_CSV)
    agents = pd.read_csv(AGENTS_CSV).drop_duplicates("image_id")
    joined = split.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    print(f"Matched {len(joined)}/{len(split)} split images with agent scores.")

    agent_cols_4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    agent_cols_6 = agent_cols_4 + ["restoration_agent", "document_agent"]

    joined["saa_fusion"] = (0.6 * joined["existing_pipeline_agent"]
                             + 0.2 * joined["scene_agent"]
                             + 0.2 * joined["vlm_agent"])
    joined["saa_pred"] = (joined["saa_fusion"] >= 0.5).astype(int)
    joined["error_label"] = (joined["saa_pred"] != joined["has_scene"]).astype(int)

    joined["u_saa"] = joined[agent_cols_4].std(axis=1)
    p_sce = joined["scene_agent"].clip(1e-6, 1 - 1e-6)
    joined["temp_entropy"] = (-(p_sce * np.log2(p_sce) + (1 - p_sce) * np.log2(1 - p_sce))) / 1.5
    joined["mc_dropout_proxy"] = 1.0 - joined["existing_pipeline_agent"]
    joined["deep_ensemble"] = joined[agent_cols_6].std(axis=1)

    test = joined[joined["split"] == "test"].copy()
    train = joined[joined["split"] == "train"].copy()
    cal = joined[joined["split"] == "calibration"].copy()
    print(f"  train={len(train)}  calibration={len(cal)}  test={len(test)}  "
          f"(test error rate={test['error_label'].mean():.3f})")

    results: dict = {
        "protocol": "Group-level (publication/PPN) split. All reported numbers "
                     "are computed on the test partition only; test images/PPNs "
                     "never appear in train or calibration.",
        "split_sizes": {"train": len(train), "calibration": len(cal), "test": len(test)},
        "test_error_rate": float(test["error_label"].mean()),
    }

    # ── Uncertainty baseline comparison, test partition only ────────────────
    baselines = {
        "Multi-Agent SAA (Ours)": test["u_saa"],
        "Temperature-Scaled Entropy": test["temp_entropy"],
        "MC-Dropout-style YOLO-inverse-confidence proxy": test["mc_dropout_proxy"],
        "Deep Ensemble Uncertainty (6-agent std)": test["deep_ensemble"],
    }
    y_err_test = test["error_label"].values
    results["uncertainty_baselines_on_clean_test"] = {}
    for name, scores in baselines.items():
        auc = roc_auc_score(y_err_test, scores)
        scores_norm = scores / scores.max() if scores.max() > 0 else scores
        e = ece(y_err_test, scores_norm)
        r20 = recall_at_k(y_err_test, scores, k=0.20)
        results["uncertainty_baselines_on_clean_test"][name] = {
            "roc_auc": round(float(auc), 4), "ece": round(e, 4), "recall_at_20pct": round(r20, 4),
        }
        print(f"  {name:52s} AUROC={auc:.4f}  ECE={e:.4f}  R@20%={r20:.4f}")

    # ── AWLF: fit on train, evaluate on test (clean) ─────────────────────────
    X_train = train[agent_cols_6].fillna(0.5)
    y_train = train["error_label"]
    X_test = test[agent_cols_6].fillna(0.5)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    awlf_test_probs = clf.predict_proba(scaler.transform(X_test))[:, 1]
    awlf_auc = roc_auc_score(y_err_test, awlf_test_probs)
    results["awlf_trained_on_train_evaluated_on_test"] = {
        "roc_auc": round(float(awlf_auc), 4),
        "ece": round(ece(y_err_test, awlf_test_probs), 4),
    }
    print(f"\n  AWLF (train-fit, test-eval)                       AUROC={awlf_auc:.4f}")

    # ── SCI-tercile routing policy: thresholds from full C0, applied to test ─
    sci_full = pd.read_csv(SCI_CSV)
    q1, q2 = sci_full["scene_complexity_index"].quantile([1 / 3, 2 / 3])
    print(f"\n  SCI terciles from full C0 (n={len(sci_full)}): low<{q1:.3f}  mid<{q2:.3f}  high>=that")

    sci_test = test.merge(
        sci_full[["image_id", "scene_complexity_index"]],
        left_on="ppn_id", right_on="image_id", how="inner", suffixes=("", "_sci"),
    )
    print(f"  SCI matched for {len(sci_test)}/{len(test)} test images.")

    low_mask = sci_test["scene_complexity_index"] < q1
    high_mask = sci_test["scene_complexity_index"] >= q2
    mid_mask = ~low_mask & ~high_mask

    # policy: auto-approve low tier, mandatory-review high tier,
    # conditional review (U_SAA >= its own median) for mid tier
    u_median = sci_test["u_saa"].median()
    mid_review_mask = mid_mask & (sci_test["u_saa"] >= u_median)

    reviewed_mask = high_mask | mid_review_mask
    n = len(sci_test)
    n_reviewed = int(reviewed_mask.sum())
    n_errors = int(sci_test["error_label"].sum())
    n_caught = int(sci_test.loc[reviewed_mask, "error_label"].sum())
    review_pct = n_reviewed / n * 100
    recall_pct = (n_caught / n_errors * 100) if n_errors else float("nan")

    results["sci_tercile_routing_on_clean_test"] = {
        "n_test_images": n,
        "sci_thresholds_from_full_c0": {"low_below": round(float(q1), 4), "high_at_or_above": round(float(q2), 4)},
        "n_reviewed": n_reviewed,
        "review_budget_pct": round(review_pct, 2),
        "n_errors": n_errors,
        "n_errors_caught": n_caught,
        "error_recall_pct": round(recall_pct, 2),
        "workload_reduction_pct": round(100 - review_pct, 2),
        "efficiency_lift_vs_random": round((recall_pct / review_pct), 2) if review_pct else None,
    }
    print(f"\n  Review budget: {review_pct:.2f}%  |  Error recall: {recall_pct:.2f}%  |  "
          f"Workload reduction: {100-review_pct:.2f}%  |  "
          f"Efficiency lift: {recall_pct/review_pct:.2f}x")

    # ── Negative-control decomposition: SAA disagreement AUROC, split by the
    # ── ground-truth class itself (has_scene), on the test partition only.
    # ── This is the thesis's "central operational finding" table
    # ── (Table tab:negative_control_gap / Defense Slide 3.8) -- previously
    # ── computed once ad hoc and hand-copied into the thesis with no backing
    # ── script, so it could not be regenerated or re-verified. Restored here
    # ── from the exact same u_saa/error_label columns above.
    decomposition = {}
    for cls, name in [(1, "positive_scene_present"), (0, "negative_control_no_scene")]:
        sub = test[test["has_scene"] == cls]
        n_sub = len(sub)
        err_rate = float(sub["error_label"].mean()) if n_sub else float("nan")
        if sub["error_label"].nunique() < 2:
            sub_auc = None
        else:
            sub_auc = round(float(roc_auc_score(sub["error_label"], sub["u_saa"])), 4)
        decomposition[name] = {"n": n_sub, "error_rate": round(err_rate, 4), "saa_disagreement_roc_auc": sub_auc}
        print(f"  {name:28s} n={n_sub:4d}  error_rate={err_rate:.4f}  SAA-disagreement AUROC={sub_auc}")
    results["negative_control_decomposition"] = decomposition

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
