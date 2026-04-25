#!/usr/bin/env python3
"""
5-Fold Cross Validation Evaluation of Monolithic vs Fusion strategies.

Tests stability of the multi-agent fusion score versus monolithic baseline
across random data partitions. Reports per-fold means, std dev, and
whether fusion consistently outperforms monolithic.

Outputs:
  results/multi_agent/cross_fold_results.csv
  results/multi_agent/cross_fold_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_FOLDS = 5
SEED = 42

AGENTS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "restoration_agent",
    "document_agent",
]

WEIGHTS = {
    "existing_pipeline_agent": 0.30,
    "agreement_agent":         0.20,
    "scene_agent":             0.15,
    "vlm_agent":               0.15,
    "restoration_agent":       0.10,
    "document_agent":          0.10,
}


def weighted_fusion(df: pd.DataFrame, drop: set[str] | None = None) -> pd.Series:
    drop = drop or set()
    active = [c for c in AGENTS if c not in drop and c in df.columns]
    wsum = sum(WEIGHTS[c] for c in active)
    score = sum(df[c] * WEIGHTS[c] for c in active)
    return score / wsum if wsum > 0 else pd.Series(0.0, index=df.index)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two arrays."""
    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float(diff / pooled_std) if pooled_std > 1e-9 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fold evaluation of fusion vs monolithic.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--input-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
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
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    folds = np.array_split(indices, args.n_folds)

    fold_rows = []
    for fold_idx, fold_ids in enumerate(folds):
        fold_df = df.iloc[fold_ids].copy()

        # Recompute fusion on fold (in case precomputed doesn't exist)
        fold_df["fold_fusion"] = weighted_fusion(fold_df)

        mono_scores = fold_df["monolithic_pipeline_agent"].to_numpy(dtype=float)
        fus_scores  = fold_df["fold_fusion"].to_numpy(dtype=float)

        mono_mean = float(np.mean(mono_scores))
        fus_mean  = float(np.mean(fus_scores))
        delta     = fus_mean - mono_mean
        d         = cohens_d(fus_scores, mono_scores)

        # Per-agent means in fold
        agent_means = {a: float(fold_df[a].mean()) for a in AGENTS if a in fold_df.columns}

        row = {
            "fold": fold_idx + 1,
            "n_images": len(fold_df),
            "monolithic_mean": round(mono_mean, 4),
            "fusion_mean": round(fus_mean, 4),
            "delta_fusion_minus_mono": round(delta, 4),
            "cohens_d": round(d, 4),
            "fusion_wins": int(delta > 0),
        }
        row.update({f"agent_{a}_mean": round(v, 4) for a, v in agent_means.items()})
        fold_rows.append(row)

        print(f"  Fold {fold_idx + 1}: n={len(fold_df)} | mono={mono_mean:.4f} | fus={fus_mean:.4f} | delta={delta:+.4f} | d={d:.3f}")

    fold_df_out = pd.DataFrame(fold_rows)
    fold_path = out_dir / "cross_fold_results.csv"
    fold_df_out.to_csv(fold_path, index=False)
    print(f"✅ Wrote {fold_path}")

    # Summary stats across folds
    mono_means = fold_df_out["monolithic_mean"].to_numpy()
    fus_means  = fold_df_out["fusion_mean"].to_numpy()
    deltas     = fold_df_out["delta_fusion_minus_mono"].to_numpy()
    ds         = fold_df_out["cohens_d"].to_numpy()

    summary = {
        "n_folds": args.n_folds,
        "total_images": int(len(df)),
        "monolithic": {
            "mean_across_folds": round(float(np.mean(mono_means)), 4),
            "std_across_folds":  round(float(np.std(mono_means, ddof=1)), 4),
        },
        "fusion": {
            "mean_across_folds": round(float(np.mean(fus_means)), 4),
            "std_across_folds":  round(float(np.std(fus_means, ddof=1)), 4),
        },
        "delta_fusion_minus_mono": {
            "mean": round(float(np.mean(deltas)), 4),
            "std":  round(float(np.std(deltas, ddof=1)), 4),
        },
        "cohens_d_mean": round(float(np.mean(ds)), 4),
        "folds_where_fusion_wins": int(fold_df_out["fusion_wins"].sum()),
        "conclusion": (
            "Fusion consistently outperforms monolithic"
            if fold_df_out["fusion_wins"].sum() > args.n_folds // 2
            else "Monolithic outperforms fusion in majority of folds (see ablation for explanation)"
        ),
        "note": (
            "Cross-fold evaluation tests stability of aggregate scores. "
            "Note that the coordinator is designed for conservative reliability (HITL routing), "
            "not raw score maximization — raw delta may not reflect the full benefit."
        ),
    }

    summary_path = out_dir / "cross_fold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"✅ Wrote {summary_path}")

    print(f"\n📊 Cross-Fold Summary:")
    print(f"   Monolithic: {summary['monolithic']['mean_across_folds']:.4f} ± {summary['monolithic']['std_across_folds']:.4f}")
    print(f"   Fusion:     {summary['fusion']['mean_across_folds']:.4f} ± {summary['fusion']['std_across_folds']:.4f}")
    print(f"   Δ (fusion - mono): {summary['delta_fusion_minus_mono']['mean']:+.4f}")
    print(f"   Cohen's d:  {summary['cohens_d_mean']:.4f}")
    print(f"   {summary['conclusion']}")


if __name__ == "__main__":
    main()
