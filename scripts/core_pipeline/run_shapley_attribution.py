#!/usr/bin/env python3
"""
Shapley Value Agent Attribution (E5).

Computes the exact Shapley value for each agent to measure its marginal
contribution across all possible coalition orderings.
This replaces the "best subset" analysis with a principled coalitional game theory metric.

The utility function v(S) is defined as the Precision@10% of identifying
"proxy issues" (or true human errors, when the gold dataset is fully populated).

Outputs:
  results/multi_agent/shapley_attribution.csv
  results/multi_agent/shapley_attribution_summary.json
"""
from __future__ import annotations
import argparse
import json
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Define the set of possible agent columns in the dataset
POTENTIAL_AGENT_COLS = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent",
    "vlm_agent", "restoration_agent", "document_agent",
]

def normalize_issue(df: pd.DataFrame) -> np.ndarray:
    """
    Creates a binary label for evaluation.
    Ideally, this will be replaced with real human labels in E1/E2.
    For now, it identifies images in the bottom 25% of either baseline or fusion score.
    """
    q1f = df["comparison_fusion_score"].quantile(0.25)
    q1m = df["monolithic_pipeline_agent"].quantile(0.25)
    return ((df["comparison_fusion_score"] <= q1f) | (df["monolithic_pipeline_agent"] <= q1m)).to_numpy(int)

def subset_precision_at_k(df: pd.DataFrame, cols: list[str], labels: np.ndarray, k=0.10) -> float:
    """
    Utility function v(S):
    Computes precision at top k% risk for a given subset of agents S.
    """
    if not cols:
        return 0.0
    
    # We use the unweighted mean of the agents in the coalition
    # This represents a simple democratic fusion for the coalition S
    subset_score = df[cols].mean(axis=1).to_numpy()
    risk = 1.0 - subset_score
    k_n = max(1, int(len(df) * k))
    
    top = np.argsort(-risk)[:k_n]
    return float(labels[top].mean())

def exact_shapley_values(agents: list[str], df: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """
    Computes exact Shapley values for the given set of agents using the utility function.
    N = len(agents)
    Formula: phi_i = sum_{S subset N \\ {i}} (|S|! (N - |S| - 1)! / N!) * (v(S U {i}) - v(S))
    """
    N = len(agents)
    shapley_values = {a: 0.0 for a in agents}
    
    # Memoize utility function to avoid recomputing v(S)
    memo = {}
    def v(S_tuple):
        if S_tuple not in memo:
            memo[S_tuple] = subset_precision_at_k(df, list(S_tuple), labels)
        return memo[S_tuple]

    for i, agent in enumerate(agents):
        # All other agents
        others = [a for a in agents if a != agent]
        
        for size in range(N):
            # All subsets of `others` of size `size`
            for subset in itertools.combinations(others, size):
                S = tuple(sorted(subset))
                S_union_i = tuple(sorted(subset + (agent,)))
                
                weight = (math.factorial(size) * math.factorial(N - size - 1)) / math.factorial(N)
                marginal_contribution = v(S_union_i) - v(S)
                
                shapley_values[agent] += weight * marginal_contribution
                
    return shapley_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    
    csv_path = base / args.scores_csv if not Path(args.scores_csv).is_absolute() else Path(args.scores_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run run_agent_comparison.py first.")
        return
        
    df = pd.read_csv(csv_path)
    out = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Determine which agents are actually in the dataset
    agents = [c for c in POTENTIAL_AGENT_COLS if c in df.columns]
    
    labels = normalize_issue(df)
    full_p = subset_precision_at_k(df, agents, labels)
    print(f"Full {len(agents)}-agent ensemble P@10% = {full_p:.4f}")
    
    print("Computing exact Shapley values (this takes a moment)...")
    shapley_vals = exact_shapley_values(agents, df, labels)
    
    # Sort by contribution
    sorted_agents = sorted(shapley_vals.items(), key=lambda x: x[1], reverse=True)
    
    rows = []
    for rank, (agent, shap_val) in enumerate(sorted_agents, 1):
        short_name = agent.replace("_agent", "").replace("_pipeline", "")
        rows.append({
            "rank": rank,
            "agent": agent,
            "short_name": short_name,
            "shapley_value": round(shap_val, 4),
            "pct_of_full_utility": round((shap_val / full_p) * 100, 2) if full_p > 0 else 0.0
        })
        
    result_df = pd.DataFrame(rows)
    out_csv = out / "shapley_attribution.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}")
    
    print("\n📊 Shapley Value Agent Attribution:")
    for row in rows:
        print(f"  {row['rank']}. {row['short_name']:<12}: {row['shapley_value']:.4f} ({row['pct_of_full_utility']:>5.1f}%)")

    summary = {
        "num_agents": len(agents),
        "agents": agents,
        "full_ensemble_precision": round(full_p, 4),
        "shapley_values": {row["agent"]: row["shapley_value"] for row in rows},
        "conclusion": (
            f"The '{rows[0]['short_name']}' agent provides the highest marginal contribution "
            f"({rows[0]['shapley_value']:.4f}), accounting for roughly {rows[0]['pct_of_full_utility']:.1f}% "
            "of the total coalition utility. Unlike simple subset analysis, these Shapley values "
            "rigorously distribute the credit of the multi-agent fusion across all possible sub-coalitions."
        )
    }

    out_json = out / "shapley_attribution_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {out_json}")

if __name__ == "__main__":
    main()
