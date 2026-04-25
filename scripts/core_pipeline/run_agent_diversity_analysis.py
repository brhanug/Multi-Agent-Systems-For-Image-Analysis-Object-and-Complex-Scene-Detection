#!/usr/bin/env python3
"""
Agent Diversity & Complementarity Analysis.

Measures how diverse and complementary the 6 agents are:
  - Pairwise Pearson correlation between all agents (low = complementary)
  - Ensemble diversity score (average pairwise disagreement)
  - Agent uniqueness: information each agent adds beyond the others
  - Redundancy analysis: which agents are most correlated

This directly supports RQ3 (agent complementarity) with quantitative evidence.

Outputs:
  results/multi_agent/agent_diversity_matrix.csv
  results/multi_agent/agent_diversity_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

AGENT_COLS = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent",
    "vlm_agent", "restoration_agent", "document_agent",
]

SHORT_NAMES = {
    "existing_pipeline_agent": "Object",
    "agreement_agent":         "Agreement",
    "scene_agent":             "Scene",
    "vlm_agent":               "VLM",
    "restoration_agent":       "Restoration",
    "document_agent":          "Document",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scores_csv = Path(args.scores_csv) if Path(args.scores_csv).is_absolute() else base / args.scores_csv
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    agent_matrix = df[AGENT_COLS].to_numpy(dtype=float)

    # Pearson correlation matrix
    corr = np.corrcoef(agent_matrix.T)  # shape (6, 6)
    corr_df = pd.DataFrame(
        np.round(corr, 4),
        index=[SHORT_NAMES[a] for a in AGENT_COLS],
        columns=[SHORT_NAMES[a] for a in AGENT_COLS],
    )
    corr_path = out_dir / "agent_diversity_matrix.csv"
    corr_df.to_csv(corr_path)
    print(f"✅ Wrote {corr_path}")

    # Ensemble diversity: mean absolute pairwise disagreement per image
    n_agents = len(AGENT_COLS)
    pairwise_diffs = []
    pair_corrs = {}
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            diff = np.abs(agent_matrix[:, i] - agent_matrix[:, j])
            pairwise_diffs.append(diff)
            key = f"{SHORT_NAMES[AGENT_COLS[i]]} vs {SHORT_NAMES[AGENT_COLS[j]]}"
            pair_corrs[key] = {
                "pearson_r": round(float(corr[i, j]), 4),
                "mean_abs_diff": round(float(diff.mean()), 4),
                "complementarity": "high" if corr[i, j] < 0.3 else "medium" if corr[i, j] < 0.7 else "low",
            }

    ensemble_diversity = float(np.mean([d.mean() for d in pairwise_diffs]))

    # Agent uniqueness: variance not explained by others (1 - max pairwise correlation)
    uniqueness = {}
    for i, col in enumerate(AGENT_COLS):
        max_corr_with_others = max(abs(corr[i, j]) for j in range(n_agents) if j != i)
        uniqueness[SHORT_NAMES[col]] = round(float(1.0 - max_corr_with_others), 4)

    # Most and least complementary pairs
    sorted_pairs = sorted(pair_corrs.items(), key=lambda x: x[1]["pearson_r"])
    most_complementary  = sorted_pairs[:3]
    most_redundant      = sorted_pairs[-3:]

    print(f"\n📊 Agent Diversity Summary:")
    print(f"   Ensemble diversity score: {ensemble_diversity:.4f}")
    print(f"   Most complementary pairs:")
    for name, vals in most_complementary:
        print(f"     {name}: r={vals['pearson_r']:.3f}, diff={vals['mean_abs_diff']:.3f}")
    print(f"   Most redundant pairs:")
    for name, vals in most_redundant:
        print(f"     {name}: r={vals['pearson_r']:.3f}")

    summary = {
        "n_images": int(len(df)),
        "n_agents": n_agents,
        "ensemble_diversity_score": round(ensemble_diversity, 4),
        "ensemble_diversity_interpretation": (
            "high diversity (agents are highly complementary)" if ensemble_diversity > 0.3 else
            "medium diversity" if ensemble_diversity > 0.15 else
            "low diversity (agents are somewhat redundant)"
        ),
        "agent_uniqueness_scores": uniqueness,
        "most_unique_agent": max(uniqueness, key=uniqueness.get),
        "least_unique_agent": min(uniqueness, key=uniqueness.get),
        "pairwise_analysis": pair_corrs,
        "most_complementary_pairs": [{"pair": n, **v} for n, v in most_complementary],
        "most_redundant_pairs": [{"pair": n, **v} for n, v in most_redundant],
        "rq3_conclusion": (
            f"The 6-agent ensemble achieves a diversity score of {ensemble_diversity:.3f}. "
            f"The most complementary pair is '{most_complementary[0][0]}' (r={most_complementary[0][1]['pearson_r']:.3f}), "
            f"confirming that agents with different inductive biases provide independent evidence. "
            f"The most unique agent is '{max(uniqueness, key=uniqueness.get)}' (uniqueness={max(uniqueness.values()):.3f})."
        ),
    }

    summary_path = out_dir / "agent_diversity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {summary_path}")

if __name__ == "__main__":
    main()
