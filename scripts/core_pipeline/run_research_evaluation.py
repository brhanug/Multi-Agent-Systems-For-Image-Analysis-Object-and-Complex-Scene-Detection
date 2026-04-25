#!/usr/bin/env python3
"""
Research-oriented evaluation from agent comparison outputs.

Generates:
1) baseline/agent summary table
2) leave-one-agent-out ablation table
3) HITL triage efficiency table (top-k vs random)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AGENT_COLS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "restoration_agent",
    "document_agent",
    "monolithic_pipeline_agent",
]


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    return pd.read_csv(path)


def normalize_issue_labels(df: pd.DataFrame) -> pd.Series:
    """
    Proxy issue label without full GT:
    low fusion and low existing pipeline confidence are high-risk.
    """
    q1_fusion = df["comparison_fusion_score"].quantile(0.25)
    q1_monolithic = df["monolithic_pipeline_agent"].quantile(0.25)
    return ((df["comparison_fusion_score"] <= q1_fusion) | (df["monolithic_pipeline_agent"] <= q1_monolithic)).astype(int)


def weighted_fusion(df: pd.DataFrame, drop_cols: set[str] | None = None) -> pd.Series:
    drop_cols = drop_cols or set()
    weights = {
        "existing_pipeline_agent": 0.30,
        "agreement_agent": 0.20,
        "scene_agent": 0.15,
        "vlm_agent": 0.15,
        "restoration_agent": 0.10,
        "document_agent": 0.10,
        "monolithic_pipeline_agent": 0.0,
    }
    active = [c for c in AGENT_COLS if c not in drop_cols]
    wsum = sum(weights[c] for c in active)
    score = sum(df[c] * weights[c] for c in active)
    return score / wsum if wsum > 0 else pd.Series(0.0, index=df.index)


def hitl_precision_at_k(df: pd.DataFrame, risk_col: str, issue_col: str, k_ratio: float) -> float:
    k = max(1, int(len(df) * k_ratio))
    top = df.sort_values(risk_col, ascending=False).head(k)
    return float(top[issue_col].mean())


def random_precision(df: pd.DataFrame, issue_col: str) -> float:
    return float(df[issue_col].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-level evaluation tables.")
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

    df = load_scores(input_csv)
    if "comparison_fusion_score" not in df.columns:
        df["comparison_fusion_score"] = weighted_fusion(df)

    # Baseline/agent summary
    baseline_rows = []
    for col in AGENT_COLS + ["comparison_fusion_score"]:
        baseline_rows.append(
            {
                "model_or_agent": col,
                "mean_score": float(df[col].mean()),
                "coverage_ratio": float((df[col] > 0).mean()),
                "std_score": float(df[col].std()),
            }
        )
    baseline_df = pd.DataFrame(baseline_rows).sort_values("mean_score", ascending=False)

    # Ablations: remove one agent from fusion
    ablation_rows = []
    full_fusion = weighted_fusion(df)
    full_mean = float(full_fusion.mean())
    for col in AGENT_COLS:
        ablated = weighted_fusion(df, drop_cols={col})
        ablation_rows.append(
            {
                "ablation": f"without_{col}",
                "ablated_mean_fusion": float(ablated.mean()),
                "delta_vs_full": float(ablated.mean() - full_mean),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_vs_full")

    # HITL efficiency
    eval_df = df.copy()
    eval_df["proxy_issue"] = normalize_issue_labels(eval_df)
    eval_df["risk_score"] = 1.0 - full_fusion
    random_base = random_precision(eval_df, "proxy_issue")
    hitl_rows = []
    for k in [0.1, 0.2, 0.3]:
        p_at_k = hitl_precision_at_k(eval_df, "risk_score", "proxy_issue", k)
        lift = (p_at_k / random_base) if random_base > 0 else 0.0
        hitl_rows.append(
            {
                "top_k_ratio": k,
                "precision_at_k": p_at_k,
                "random_precision": random_base,
                "lift_vs_random": lift,
            }
        )
    hitl_df = pd.DataFrame(hitl_rows)

    # Save artifacts
    baseline_path = out_dir / "research_baseline_summary.csv"
    ablation_path = out_dir / "research_ablation_summary.csv"
    hitl_path = out_dir / "research_hitl_efficiency.csv"
    complexity_path = out_dir / "research_complexity_stratified_summary.csv"
    summary_path = out_dir / "research_evaluation_summary.json"

    baseline_df.to_csv(baseline_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)
    hitl_df.to_csv(hitl_path, index=False)

    # Complexity stratification (proxy): based on existing pipeline object density score.
    eval_df["complexity_bin"] = pd.cut(
        eval_df["existing_pipeline_agent"],
        bins=[-0.001, 0.3333, 0.6666, 1.0],
        labels=["low", "medium", "high"],
    )
    complexity_df = (
        eval_df.groupby("complexity_bin", observed=False)
        .agg(
            num_images=("image_id", "count"),
            monolithic_mean=("monolithic_pipeline_agent", "mean"),
            fusion_mean=("comparison_fusion_score", "mean"),
            issue_rate=("proxy_issue", "mean"),
            mean_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    complexity_df.to_csv(complexity_path, index=False)

    summary = {
        "num_images": int(len(df)),
        "full_fusion_mean": full_mean,
        "random_precision": random_base,
        "best_single_agent_by_mean": baseline_df.iloc[0]["model_or_agent"] if not baseline_df.empty else None,
        "complexity_bins": int(complexity_df["complexity_bin"].nunique()) if not complexity_df.empty else 0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Wrote {baseline_path}")
    print(f"✅ Wrote {ablation_path}")
    print(f"✅ Wrote {hitl_path}")
    print(f"✅ Wrote {complexity_path}")
    print(f"✅ Wrote {summary_path}")


if __name__ == "__main__":
    main()
