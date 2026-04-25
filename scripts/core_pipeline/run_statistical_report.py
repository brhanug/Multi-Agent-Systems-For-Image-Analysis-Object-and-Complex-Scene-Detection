#!/usr/bin/env python3
"""
Generate statistical report with bootstrap confidence intervals
and paired significance-style comparisons for agent scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AGENTS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "restoration_agent",
    "document_agent",
    "monolithic_pipeline_agent",
    "comparison_fusion_score",
]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(float(np.mean(values[idx])))
    means_arr = np.array(means, dtype=float)
    low = float(np.quantile(means_arr, alpha / 2))
    high = float(np.quantile(means_arr, 1 - alpha / 2))
    return float(np.mean(values)), low, high


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float, bool]:
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0, False
    a = a[:n]
    b = b[:n]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(float(np.mean(a[idx] - b[idx])))
    arr = np.array(deltas, dtype=float)
    low = float(np.quantile(arr, alpha / 2))
    high = float(np.quantile(arr, 1 - alpha / 2))
    delta = float(np.mean(a - b))
    significant = (low > 0) or (high < 0)
    return delta, low, high, significant


def main() -> None:
    parser = argparse.ArgumentParser(description="Create statistical report for agent comparisons.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--input-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = base / input_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    for col in AGENTS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    ci_rows = []
    for col in AGENTS:
        mean, ci_low, ci_high = bootstrap_mean_ci(df[col].to_numpy(dtype=float), n_boot=args.bootstrap_samples)
        ci_rows.append(
            {
                "metric": col,
                "mean": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "coverage_ratio": float((df[col] > 0).mean()),
            }
        )
    ci_df = pd.DataFrame(ci_rows).sort_values("mean", ascending=False)

    # Compare each metric to existing pipeline and to fusion
    diff_rows = []
    existing = df["existing_pipeline_agent"].to_numpy(dtype=float)
    fusion = df["comparison_fusion_score"].to_numpy(dtype=float)
    for col in AGENTS:
        arr = df[col].to_numpy(dtype=float)
        d1, l1, h1, s1 = bootstrap_delta_ci(arr, existing, n_boot=args.bootstrap_samples)
        d2, l2, h2, s2 = bootstrap_delta_ci(arr, fusion, n_boot=args.bootstrap_samples)
        diff_rows.append(
            {
                "metric": col,
                "delta_vs_existing": d1,
                "delta_vs_existing_ci95_low": l1,
                "delta_vs_existing_ci95_high": h1,
                "delta_vs_existing_significant": int(s1),
                "delta_vs_fusion": d2,
                "delta_vs_fusion_ci95_low": l2,
                "delta_vs_fusion_ci95_high": h2,
                "delta_vs_fusion_significant": int(s2),
            }
        )
    diff_df = pd.DataFrame(diff_rows)

    ci_path = out_dir / "statistical_ci_summary.csv"
    diff_path = out_dir / "statistical_pairwise_deltas.csv"
    summary_path = out_dir / "statistical_report_summary.json"
    ci_df.to_csv(ci_path, index=False)
    diff_df.to_csv(diff_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_images": int(len(df)),
                "bootstrap_samples": int(args.bootstrap_samples),
                "best_metric_by_mean": ci_df.iloc[0]["metric"] if not ci_df.empty else None,
            },
            f,
            indent=2,
        )

    print(f"✅ Wrote {ci_path}")
    print(f"✅ Wrote {diff_path}")
    print(f"✅ Wrote {summary_path}")


if __name__ == "__main__":
    main()
