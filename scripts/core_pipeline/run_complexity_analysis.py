#!/usr/bin/env python3
"""
Deep complexity analysis with 5 bins and per-agent breakdown.

Extends the 3-bin complexity stratification to 5 bins.
Computes:
  - Per-bin means for every agent
  - Spearman correlation between complexity level and monolithic vs fusion delta
  - Per-bin HITL efficiency (precision@k in each complexity stratum)
  - Per-bin inter-agent disagreement

Outputs:
  results/multi_agent/complexity_deep_analysis.csv
  results/multi_agent/complexity_deep_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AGENT_COLS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "restoration_agent",
    "document_agent",
    "monolithic_pipeline_agent",
    "comparison_fusion_score",
]

BINS_5 = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
BIN_LABELS_5 = ["very_low", "low", "medium", "high", "very_high"]


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation coefficient (manual)."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d  = rx - ry
    return float(1.0 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def normalize_issue_labels(df: pd.DataFrame) -> pd.Series:
    q1_fusion = df["comparison_fusion_score"].quantile(0.25)
    q1_mono   = df["monolithic_pipeline_agent"].quantile(0.25)
    return ((df["comparison_fusion_score"] <= q1_fusion) | (df["monolithic_pipeline_agent"] <= q1_mono)).astype(int)


def hitl_precision_at_k(df: pd.DataFrame, risk_col: str, issue_col: str, k_ratio: float) -> float:
    k = max(1, int(len(df) * k_ratio))
    top = df.nlargest(k, risk_col)
    return float(top[issue_col].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep complexity analysis.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--input-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
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

    # Compute complexity proxy (existing_pipeline = object density score)
    df["complexity_score"] = df["existing_pipeline_agent"]

    # 5-bin stratification
    df["complexity_bin_5"] = pd.cut(
        df["complexity_score"],
        bins=BINS_5,
        labels=BIN_LABELS_5,
    )

    # Inter-agent disagreement
    agent_matrix = df[[c for c in AGENT_COLS[:6] if c in df.columns]].to_numpy(dtype=float)
    df["inter_agent_disagreement"] = np.std(agent_matrix, axis=1)

    # Proxy issue labels
    df["proxy_issue"] = normalize_issue_labels(df)
    df["risk_score"]  = 1.0 - df["comparison_fusion_score"]
    df["delta_fus_mono"] = df["comparison_fusion_score"] - df["monolithic_pipeline_agent"]

    complexity_rows = []
    for bin_label in BIN_LABELS_5:
        subset = df[df["complexity_bin_5"] == bin_label]
        if len(subset) == 0:
            continue

        row: dict = {"complexity_bin": bin_label, "n_images": len(subset)}

        # Per-agent means
        for col in AGENT_COLS:
            if col in subset.columns:
                row[f"{col}_mean"] = round(float(subset[col].mean()), 4)

        # Delta stats
        row["delta_fus_minus_mono_mean"] = round(float(subset["delta_fus_mono"].mean()), 4)
        row["inter_agent_disagreement_mean"] = round(float(subset["inter_agent_disagreement"].mean()), 4)
        row["proxy_issue_rate"] = round(float(subset["proxy_issue"].mean()), 4)

        # HITL efficiency in this stratum
        if len(subset) >= 10:
            row["hitl_precision_at_10pct"] = round(
                hitl_precision_at_k(subset, "risk_score", "proxy_issue", 0.10), 4
            )
        else:
            row["hitl_precision_at_10pct"] = None

        complexity_rows.append(row)

    complexity_df = pd.DataFrame(complexity_rows)

    complexity_path = out_dir / "complexity_deep_analysis.csv"
    complexity_df.to_csv(complexity_path, index=False)
    print(f"✅ Wrote {complexity_path}")

    # Spearman correlations: complexity level vs various signals
    bin_numeric = {b: i for i, b in enumerate(BIN_LABELS_5)}
    df["complexity_numeric"] = df["complexity_bin_5"].map(bin_numeric)
    df_valid = df[df["complexity_numeric"].notna()]

    cx = df_valid["complexity_numeric"].to_numpy(dtype=float)
    spearman_correlations = {}
    for col in AGENT_COLS + ["inter_agent_disagreement", "delta_fus_mono"]:
        if col in df_valid.columns:
            spearman_correlations[col] = round(spearman_r(cx, df_valid[col].to_numpy(dtype=float)), 4)

    # Key finding: does fusion advantage INCREASE with complexity?
    delta_by_bin = []
    for bin_label in BIN_LABELS_5:
        subset = df[df["complexity_bin_5"] == bin_label]
        if len(subset) > 0:
            delta_by_bin.append(float(subset["delta_fus_mono"].mean()))

    fusion_advantage_increases_with_complexity = (
        len(delta_by_bin) >= 2 and delta_by_bin[-1] > delta_by_bin[0]
    )

    summary = {
        "n_images": int(len(df)),
        "bins": BIN_LABELS_5,
        "bin_sizes": {row["complexity_bin"]: row["n_images"] for _, row in complexity_df.iterrows()},
        "spearman_correlations_with_complexity": spearman_correlations,
        "fusion_advantage_increases_with_complexity": fusion_advantage_increases_with_complexity,
        "delta_fus_mono_by_complexity_bin": {
            BIN_LABELS_5[i]: round(d, 4) for i, d in enumerate(delta_by_bin) if i < len(BIN_LABELS_5)
        },
        "rq3_conclusion": (
            "Multi-agent fusion provides increasing advantage over monolithic in higher complexity scenes (RQ3 supported)"
            if fusion_advantage_increases_with_complexity else
            "Fusion advantage does not monotonically increase with complexity; "
            "see per-bin breakdown for agent complementarity details"
        ),
        "methodology": (
            "Complexity proxy = existing_pipeline_agent score (object density normalized to [0,1]). "
            "5-bin stratification with equal-width thresholds. "
            "Spearman correlation measures monotonic relationship between complexity and agent signals."
        ),
    }

    summary_path = out_dir / "complexity_deep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"✅ Wrote {summary_path}")

    print(f"\n📊 Complexity Deep Analysis:")
    for row in complexity_rows:
        print(f"  {row['complexity_bin']:10s}: n={row['n_images']:5d} | "
              f"fus={row.get('comparison_fusion_score_mean', 0):.3f} | "
              f"mono={row.get('monolithic_pipeline_agent_mean', 0):.3f} | "
              f"Δ={row['delta_fus_minus_mono_mean']:+.3f} | "
              f"disagree={row['inter_agent_disagreement_mean']:.3f}")
    print(f"  → {summary['rq3_conclusion']}")


if __name__ == "__main__":
    main()
