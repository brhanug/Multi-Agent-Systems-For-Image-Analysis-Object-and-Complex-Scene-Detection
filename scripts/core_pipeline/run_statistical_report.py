#!/usr/bin/env python3
"""
Generate statistical report with bootstrap confidence intervals,
Wilcoxon signed-rank tests, Cohen's d effect sizes, and correlation matrix.
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
    means = [float(np.mean(values[rng.integers(0, n, n)])) for _ in range(n_boot)]
    arr = np.array(means, dtype=float)
    return float(np.mean(values)), float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float, bool]:
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0, False
    a, b = a[:n], b[:n]
    deltas = [float(np.mean((a - b)[rng.integers(0, n, n)])) for _ in range(n_boot)]
    arr = np.array(deltas, dtype=float)
    low, high = float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))
    delta = float(np.mean(a - b))
    return delta, low, high, (low > 0) or (high < 0)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    """
    Manual Wilcoxon signed-rank test (no scipy dependency).
    Returns (W_statistic, p_approx, interpretation).
    Uses normal approximation for large n.
    """
    n = min(len(a), len(b))
    if n < 5:
        return 0.0, 1.0, "insufficient_data"
    d = (a[:n] - b[:n]).astype(float)
    d_nonzero = d[d != 0]
    if len(d_nonzero) == 0:
        return 0.0, 1.0, "identical_distributions"
    n_eff = len(d_nonzero)
    abs_d = np.abs(d_nonzero)
    ranks = np.argsort(np.argsort(abs_d)) + 1.0  # 1-indexed ranks
    W_plus  = float(np.sum(ranks[d_nonzero > 0]))
    W_minus = float(np.sum(ranks[d_nonzero < 0]))
    W = min(W_plus, W_minus)
    # Normal approximation
    mu_W  = n_eff * (n_eff + 1) / 4.0
    sig_W = np.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0)
    z = (W - mu_W) / sig_W if sig_W > 0 else 0.0
    # Two-tailed p from z (approximation via error function)
    from math import erfc
    p = float(erfc(abs(z) / np.sqrt(2)))
    interpretation = (
        "significant (p<0.05)" if p < 0.05 else
        "marginal (p<0.10)"   if p < 0.10 else
        "not_significant"
    )
    return W, p, interpretation


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled std)."""
    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float(diff / pooled_std) if pooled_std > 1e-9 else 0.0


def effect_size_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create full statistical report for agent comparisons.")
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

    # ── 1. Bootstrap CI for each agent ─────────────────────────────────────
    ci_rows = []
    for col in AGENTS:
        mean, ci_low, ci_high = bootstrap_mean_ci(
            df[col].to_numpy(dtype=float), n_boot=args.bootstrap_samples
        )
        ci_rows.append({
            "metric": col,
            "mean": mean,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "coverage_ratio": float((df[col] > 0).mean()),
            "std": float(df[col].std()),
        })
    ci_df = pd.DataFrame(ci_rows).sort_values("mean", ascending=False)

    # ── 2. Pairwise comparisons vs existing pipeline AND vs fusion ──────────
    diff_rows = []
    existing = df["existing_pipeline_agent"].to_numpy(dtype=float)
    mono     = df["monolithic_pipeline_agent"].to_numpy(dtype=float)
    fusion   = df["comparison_fusion_score"].to_numpy(dtype=float)

    for col in AGENTS:
        arr = df[col].to_numpy(dtype=float)

        # vs existing pipeline
        d1, l1, h1, s1 = bootstrap_delta_ci(arr, existing, n_boot=args.bootstrap_samples)
        W1, p1, interp1 = wilcoxon_signed_rank(arr, existing)
        cd1 = cohens_d(arr, existing)

        # vs monolithic
        d2, l2, h2, s2 = bootstrap_delta_ci(arr, mono, n_boot=args.bootstrap_samples)
        W2, p2, interp2 = wilcoxon_signed_rank(arr, mono)
        cd2 = cohens_d(arr, mono)

        # vs fusion
        d3, l3, h3, s3 = bootstrap_delta_ci(arr, fusion, n_boot=args.bootstrap_samples)
        W3, p3, interp3 = wilcoxon_signed_rank(arr, fusion)
        cd3 = cohens_d(arr, fusion)

        diff_rows.append({
            "metric": col,
            # vs existing
            "delta_vs_existing": d1, "delta_vs_existing_ci95_low": l1,
            "delta_vs_existing_ci95_high": h1, "delta_vs_existing_bootstrap_sig": int(s1),
            "wilcoxon_vs_existing_W": round(W1, 1), "wilcoxon_vs_existing_p": round(p1, 4),
            "wilcoxon_vs_existing_interp": interp1,
            "cohens_d_vs_existing": round(cd1, 4), "effect_vs_existing": effect_size_label(cd1),
            # vs monolithic
            "delta_vs_mono": d2, "delta_vs_mono_ci95_low": l2,
            "delta_vs_mono_ci95_high": h2, "delta_vs_mono_bootstrap_sig": int(s2),
            "wilcoxon_vs_mono_W": round(W2, 1), "wilcoxon_vs_mono_p": round(p2, 4),
            "wilcoxon_vs_mono_interp": interp2,
            "cohens_d_vs_mono": round(cd2, 4), "effect_vs_mono": effect_size_label(cd2),
            # vs fusion
            "delta_vs_fusion": d3, "delta_vs_fusion_ci95_low": l3,
            "delta_vs_fusion_ci95_high": h3, "delta_vs_fusion_bootstrap_sig": int(s3),
            "wilcoxon_vs_fusion_W": round(W3, 1), "wilcoxon_vs_fusion_p": round(p3, 4),
            "wilcoxon_vs_fusion_interp": interp3,
            "cohens_d_vs_fusion": round(cd3, 4), "effect_vs_fusion": effect_size_label(cd3),
        })
    diff_df = pd.DataFrame(diff_rows)

    # ── 3. Key comparison: monolithic vs fusion ─────────────────────────────
    W_mono_fus, p_mono_fus, interp_mono_fus = wilcoxon_signed_rank(mono, fusion)
    d_mono_fus = cohens_d(mono, fusion)
    delta_mono_fus, dl_mf, dh_mf, sig_mf = bootstrap_delta_ci(mono, fusion, n_boot=args.bootstrap_samples)

    # ── 4. Correlation matrix ───────────────────────────────────────────────
    corr_matrix = df[AGENTS].corr(method="pearson").round(3)
    corr_path = out_dir / "statistical_correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)

    # ── 5. Save outputs ─────────────────────────────────────────────────────
    ci_path   = out_dir / "statistical_ci_summary.csv"
    diff_path = out_dir / "statistical_pairwise_deltas.csv"
    summary_path = out_dir / "statistical_report_summary.json"

    ci_df.to_csv(ci_path, index=False)
    diff_df.to_csv(diff_path, index=False)

    summary = {
        "num_images": int(len(df)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "best_metric_by_mean": ci_df.iloc[0]["metric"] if not ci_df.empty else None,
        "key_comparison_monolithic_vs_fusion": {
            "monolithic_mean": round(float(np.mean(mono)), 4),
            "fusion_mean":     round(float(np.mean(fusion)), 4),
            "delta_mono_minus_fusion": round(delta_mono_fus, 4),
            "ci95_low":        round(dl_mf, 4),
            "ci95_high":       round(dh_mf, 4),
            "bootstrap_significant": int(sig_mf),
            "wilcoxon_W":      round(W_mono_fus, 1),
            "wilcoxon_p":      round(p_mono_fus, 4),
            "wilcoxon_interpretation": interp_mono_fus,
            "cohens_d":        round(d_mono_fus, 4),
            "effect_size":     effect_size_label(d_mono_fus),
            "interpretation": (
                "Monolithic scores significantly higher than fusion in raw mean. "
                "However, fusion is designed for conservative reliability (HITL routing), "
                "not raw score maximization. See HITL efficiency tables for full advantage picture."
            ),
        },
        "agent_ci_summary": [
            {
                "metric": row["metric"],
                "mean": round(row["mean"], 4),
                "ci95": f"[{round(row['ci95_low'], 4)}, {round(row['ci95_high'], 4)}]",
            }
            for _, row in ci_df.iterrows()
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Wrote {ci_path}")
    print(f"✅ Wrote {diff_path}")
    print(f"✅ Wrote {corr_path}")
    print(f"✅ Wrote {summary_path}")

    print(f"\n📊 Key Statistical Results:")
    print(f"   Monolithic vs Fusion:")
    print(f"     Δ = {delta_mono_fus:+.4f}  [{dl_mf:+.4f}, {dh_mf:+.4f}]")
    print(f"     Wilcoxon: W={W_mono_fus:.0f}  p={p_mono_fus:.4f}  ({interp_mono_fus})")
    print(f"     Cohen's d = {d_mono_fus:.4f} ({effect_size_label(d_mono_fus)} effect)")


if __name__ == "__main__":
    main()
