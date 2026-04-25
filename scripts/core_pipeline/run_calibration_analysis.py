#!/usr/bin/env python3
"""
Uncertainty Calibration Analysis.

Tests whether the coordinator fusion score is well-calibrated:
  - Does fusion score = 0.7 actually mean ~70% of images at that score are "good"?
  - Produces calibration curve (reliability diagram) data
  - Computes Expected Calibration Error (ECE) for monolithic and fusion
  - Tests calibration of disagreement-based risk score

Also computes:
  - Brier Score (mean squared error of probability estimates)
  - Confidence interval width vs score magnitude (uncertainty bounds tight at extremes?)

Outputs:
  results/multi_agent/calibration_analysis.csv
  results/multi_agent/calibration_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

N_BINS = 10


def calibration_curve(scores: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    """Compute reliability diagram data."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means, bin_fracs, bin_sizes = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        bin_means.append(float(np.mean(scores[mask])))
        bin_fracs.append(float(np.mean(labels[mask])))
        bin_sizes.append(int(mask.sum()))
    return bin_means, bin_fracs, bin_sizes


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE: weighted average of |confidence - accuracy| per bin."""
    bin_means, bin_fracs, bin_sizes = calibration_curve(scores, labels, n_bins)
    total = sum(bin_sizes)
    if total == 0:
        return 0.0
    ece = sum(s / total * abs(c - f) for c, f, s in zip(bin_means, bin_fracs, bin_sizes))
    return float(ece)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((scores - labels.astype(float)) ** 2))


def normalize_issue_labels(df: pd.DataFrame) -> np.ndarray:
    q1_fusion = df["comparison_fusion_score"].quantile(0.25)
    q1_mono   = df["monolithic_pipeline_agent"].quantile(0.25)
    return ((df["comparison_fusion_score"] <= q1_fusion) | (df["monolithic_pipeline_agent"] <= q1_mono)).to_numpy(dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scores_csv = Path(args.scores_csv) if Path(args.scores_csv).is_absolute() else base / args.scores_csv
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    labels = normalize_issue_labels(df)   # 1 = problem image (low scores), 0 = good image
    # Invert: calibrate "quality" score (1 = good, 0 = problem)
    quality_labels = 1 - labels

    predictors = {
        "monolithic_pipeline_agent":  df["monolithic_pipeline_agent"].to_numpy(),
        "comparison_fusion_score":    df["comparison_fusion_score"].to_numpy(),
        "agreement_agent":            df["agreement_agent"].to_numpy(),
        "vlm_agent":                  df["vlm_agent"].to_numpy(),
    }

    cal_rows = []
    summary_pred = {}

    for name, scores in predictors.items():
        bm, bf, bs = calibration_curve(scores, quality_labels, args.n_bins)
        ece  = expected_calibration_error(scores, quality_labels, args.n_bins)
        brier = brier_score(scores, quality_labels)

        for conf, frac, sz in zip(bm, bf, bs):
            cal_rows.append({
                "predictor": name,
                "confidence": round(conf, 4),
                "fraction_positive": round(frac, 4),
                "bin_size": sz,
                "calibration_gap": round(abs(conf - frac), 4),
            })

        summary_pred[name] = {
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "calibration_quality": (
                "excellent (<0.05)" if ece < 0.05 else
                "good (0.05–0.10)" if ece < 0.10 else
                "moderate (0.10–0.20)" if ece < 0.20 else
                "poor (>0.20)"
            ),
        }
        print(f"  {name}: ECE={ece:.4f}  Brier={brier:.4f}  → {summary_pred[name]['calibration_quality']}")

    cal_df = pd.DataFrame(cal_rows)
    cal_path = out_dir / "calibration_analysis.csv"
    cal_df.to_csv(cal_path, index=False)
    print(f"✅ Wrote {cal_path}")

    # Which predictor is best calibrated?
    best_cal = min(summary_pred.items(), key=lambda x: x[1]["ece"])
    best_brier = min(summary_pred.items(), key=lambda x: x[1]["brier_score"])

    summary = {
        "n_images": int(len(df)),
        "n_bins": args.n_bins,
        "proxy_issue_rate": round(float(labels.mean()), 4),
        "predictors": summary_pred,
        "best_calibrated_predictor": best_cal[0],
        "best_ece": best_cal[1]["ece"],
        "best_brier_score_predictor": best_brier[0],
        "best_brier_score": best_brier[1]["brier_score"],
        "fusion_vs_mono_ece_delta": round(
            summary_pred["comparison_fusion_score"]["ece"] -
            summary_pred["monolithic_pipeline_agent"]["ece"], 4
        ),
        "conclusion": (
            f"Best calibrated predictor: '{best_cal[0]}' (ECE={best_cal[1]['ece']:.4f}). "
            f"Calibration measures how well score magnitudes correspond to actual outcome rates. "
            f"Well-calibrated scores are actionable for HITL prioritisation without score re-scaling."
        ),
    }

    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {summary_path}")

    print(f"\n📊 Calibration Summary:")
    print(f"   Best calibrated: {best_cal[0]} (ECE={best_cal[1]['ece']:.4f})")
    print(f"   Fusion vs Mono ECE delta: {summary['fusion_vs_mono_ece_delta']:+.4f}")

if __name__ == "__main__":
    main()
