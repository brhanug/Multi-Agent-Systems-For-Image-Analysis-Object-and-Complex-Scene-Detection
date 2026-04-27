#!/usr/bin/env python3
"""
SAA Metric Ablation Study.

Compares 4 agreement signal variants as quality/uncertainty predictors:
  1. SAA_full      : current agreement_agent (proxy for full SAA)
  2. object_only   : existing_pipeline_agent (object detection only)
  3. scene_only    : scene_agent (scene classification only)
  4. vote_majority : hard majority vote across all 6 agents (>0.5 threshold)
  5. mean_ensemble : simple unweighted average of all 6 agents

For each variant, evaluates:
  - HITL precision@10% vs proxy issue labels
  - Bootstrap 95% CI of mean
  - Correlation with proxy issue label
  - ECE calibration

This directly answers: "Why SAA? Why not simpler alternatives?" (RQ5)

Outputs:
  results/multi_agent/saa_ablation_results.csv
  results/multi_agent/saa_ablation_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

AGENT_COLS_6 = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent",
    "vlm_agent", "restoration_agent", "document_agent",
]

def normalize_issue_labels(df: pd.DataFrame) -> np.ndarray:
    q1f = df["comparison_fusion_score"].quantile(0.25)
    q1m = df["monolithic_pipeline_agent"].quantile(0.25)
    return ((df["comparison_fusion_score"] <= q1f) | (df["monolithic_pipeline_agent"] <= q1m)).to_numpy(int)

def bootstrap_ci(vals: np.ndarray, n_boot=1000, seed=42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = [float(np.mean(vals[rng.integers(0,n,n)])) for _ in range(n_boot)]
    arr = np.array(means)
    return float(np.mean(vals)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

def precision_at_k(risk: np.ndarray, labels: np.ndarray, k_ratio: float) -> float:
    k = max(1, int(len(risk)*k_ratio))
    top_idx = np.argsort(-risk)[:k]
    return float(labels[top_idx].mean())

def ece(scores: np.ndarray, labels: np.ndarray, n_bins=10) -> float:
    bins = np.linspace(0,1,n_bins+1)
    total, weighted = len(scores), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0: continue
        weighted += mask.sum() / total * abs(float(np.mean(scores[mask])) - float(np.mean(labels[mask])))
    return weighted

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm*ym).sum() / denom) if denom > 1e-9 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",  default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv",default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir",default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    df   = pd.read_csv(base / args.scores_csv if not Path(args.scores_csv).is_absolute() else args.scores_csv)
    out  = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = normalize_issue_labels(df)
    am     = df[[c for c in AGENT_COLS_6 if c in df.columns]].to_numpy(dtype=float)

    # Build signal variants
    vote_majority  = (am > 0.5).mean(axis=1)         # fraction of agents voting "good"
    mean_ensemble  = am.mean(axis=1)

    variants = {
        "SAA_agreement_agent":   df["agreement_agent"].to_numpy(dtype=float),
        "object_only":           df["existing_pipeline_agent"].to_numpy(dtype=float),
        "scene_only":            df["scene_agent"].to_numpy(dtype=float),
        "vlm_only":              df["vlm_agent"].to_numpy(dtype=float),
        "vote_majority":         vote_majority,
        "mean_ensemble":         mean_ensemble,
        "coordinator_fusion":    df["comparison_fusion_score"].to_numpy(dtype=float),
        "monolithic_baseline":   df["monolithic_pipeline_agent"].to_numpy(dtype=float),
    }

    rows = []
    for name, sig in variants.items():
        risk  = 1.0 - sig   # higher risk = lower quality signal
        mean, ci_lo, ci_hi = bootstrap_ci(sig)
        p_at_k  = precision_at_k(risk, labels, 0.10)
        ece_val = ece(sig, 1 - labels)           # quality labels (0=problem, 1=good)
        r_corr  = pearson_r(sig, 1 - labels.astype(float))  # positive r = signal predicts quality

        rows.append({
            "variant":             name,
            "mean":                round(mean, 4),
            "ci95_low":            round(ci_lo, 4),
            "ci95_high":           round(ci_hi, 4),
            "precision_at_10pct":  round(p_at_k, 4),
            "ece":                 round(ece_val, 4),
            "pearson_r_quality":   round(r_corr, 4),
        })
        print(f"  {name:<28} mean={mean:.4f} P@10%={p_at_k:.4f} ECE={ece_val:.4f} r={r_corr:.4f}")

    result_df = pd.DataFrame(rows)
    csv_path  = out / "saa_ablation_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"✅ Wrote {csv_path}")

    # Rank by precision@10%
    best_p  = result_df.loc[result_df["precision_at_10pct"].idxmax()]
    best_ece = result_df.loc[result_df["ece"].idxmin()]

    summary = {
        "n_images":                int(len(df)),
        "results":                 rows,
        "best_precision_at_10pct": {"variant": best_p["variant"], "value": float(best_p["precision_at_10pct"])},
        "best_calibration_ece":    {"variant": best_ece["variant"], "value": float(best_ece["ece"])},
        "saa_vs_object_only_delta_p": round(
            float(result_df[result_df["variant"]=="SAA_agreement_agent"]["precision_at_10pct"].iloc[0]) -
            float(result_df[result_df["variant"]=="object_only"]["precision_at_10pct"].iloc[0]), 4
        ),
        "saa_vs_vote_majority_delta_p": round(
            float(result_df[result_df["variant"]=="SAA_agreement_agent"]["precision_at_10pct"].iloc[0]) -
            float(result_df[result_df["variant"]=="vote_majority"]["precision_at_10pct"].iloc[0]), 4
        ),
        "rq5_conclusion": (
            f"SAA agreement agent achieves precision@10% = "
            f"{result_df[result_df['variant']=='SAA_agreement_agent']['precision_at_10pct'].iloc[0]:.4f}. "
            f"Compared to: object-only={result_df[result_df['variant']=='object_only']['precision_at_10pct'].iloc[0]:.4f}, "
            f"vote-majority={result_df[result_df['variant']=='vote_majority']['precision_at_10pct'].iloc[0]:.4f}, "
            f"mean-ensemble={result_df[result_df['variant']=='mean_ensemble']['precision_at_10pct'].iloc[0]:.4f}. "
            f"Best overall predictor: {best_p['variant']} ({float(best_p['precision_at_10pct']):.4f})."
        ),
    }

    sum_path = out / "saa_ablation_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {sum_path}")
    print(f"\n📊 RQ5 (SAA ablation) — Best P@10%: {best_p['variant']} ({float(best_p['precision_at_10pct']):.4f})")

if __name__ == "__main__":
    main()
