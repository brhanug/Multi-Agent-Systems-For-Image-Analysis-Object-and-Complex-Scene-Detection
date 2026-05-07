#!/usr/bin/env python3
"""
run_shapley_attribution.py
---------------------------
E5: Exact Shapley value attribution for the 4 validation agents.
N=4 → 16 coalitions → exact enumeration, no external library needed.

Validation agents only (Document + Restoration are enrichment and excluded):
    object    → existing_pipeline_agent
    agreement → agreement_agent
    scene     → scene_agent
    vlm       → vlm_agent

Characteristic function v(S):
    Mean over all images of the equal-weight average score for agents in S.

Outputs:
    results/multi_agent/shapley_attribution.json
    results/multi_agent/shapley_attribution.csv
"""

from __future__ import annotations
import itertools, json, math
from pathlib import Path
import pandas as pd

BASE       = Path(__file__).resolve().parents[2]
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
OUTPUT_DIR = BASE / "results" / "multi_agent"

AGENTS = {
    "object":    "existing_pipeline_agent",
    "agreement": "agreement_agent",
    "scene":     "scene_agent",
    "vlm":       "vlm_agent",
}
AGENT_NAMES = list(AGENTS.keys())
N = len(AGENT_NAMES)


def v(df: pd.DataFrame, coalition: tuple) -> float:
    if not coalition:
        return 0.0
    cols = [AGENTS[a] for a in coalition if AGENTS[a] in df.columns]
    return float(df[cols].mean(axis=1).mean()) if cols else 0.0


def shapley(df: pd.DataFrame) -> dict[str, float]:
    result = {a: 0.0 for a in AGENT_NAMES}
    for agent in AGENT_NAMES:
        others = [a for a in AGENT_NAMES if a != agent]
        phi = 0.0
        for size in range(N):
            for S in itertools.combinations(others, size):
                w = math.factorial(size) * math.factorial(N - size - 1) / math.factorial(N)
                phi += w * (v(df, S + (agent,)) - v(df, S))
        result[agent] = phi
    return result


def main() -> None:
    if not SCORES_CSV.exists():
        print(f"❌ Not found: {SCORES_CSV}"); return

    df = pd.read_csv(SCORES_CSV).drop_duplicates("image_id")
    print(f"📂 {len(df)} unique images")

    missing = [c for c in AGENTS.values() if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}"); return

    # All 16 coalitions
    coalition_rows = []
    for size in range(N + 1):
        for S in itertools.combinations(AGENT_NAMES, size):
            coalition_rows.append({"coalition": "+".join(S) or "∅", "size": size, "v_S": round(v(df, S), 6)})
    c_df = pd.DataFrame(coalition_rows).sort_values(["size", "coalition"])
    print("\nCoalition values:\n", c_df.to_string(index=False))

    shap = shapley(df)
    total = sum(shap.values())
    v_grand = v(df, tuple(AGENT_NAMES))

    print(f"\n{'Agent':12s} | {'φ':10s} | {'%':8s}")
    print("-" * 38)
    for a, phi in sorted(shap.items(), key=lambda x: -x[1]):
        print(f"{a:12s} | {phi:.6f} | {phi/total*100:.2f}%")
    print(f"{'Sum':12s} | {total:.6f}")
    print(f"v(grand) = {v_grand:.6f}  |  Δ = {abs(total - v_grand):.2e}")

    print("\nMarginal (leave-one-out):")
    for a in AGENT_NAMES:
        loo = v(df, tuple(x for x in AGENT_NAMES if x != a))
        print(f"  Remove {a:12s}: v = {loo:.6f}  Δ = {v_grand - loo:+.6f}")

    out_json = OUTPUT_DIR / "shapley_attribution.json"
    out_csv  = OUTPUT_DIR / "shapley_attribution.csv"
    with open(out_json, "w") as f:
        json.dump({
            "n_agents": N, "n_images": len(df),
            "grand_coalition_value": round(v_grand, 6),
            "shapley_values": {k: round(v2, 6) for k, v2 in shap.items()},
            "shapley_sum": round(total, 6),
            "efficiency_delta": round(abs(total - v_grand), 8),
            "ranking": sorted(shap, key=lambda k: -shap[k]),
            "coalition_table": c_df.to_dict(orient="records"),
        }, f, indent=2)
    pd.DataFrame([{"agent": k, "shapley_value": round(v2, 6),
                   "share_pct": round(v2/total*100, 2)}
                  for k, v2 in sorted(shap.items(), key=lambda x: -x[1])
                  ]).to_csv(out_csv, index=False)

    print(f"\n✅ {out_json}\n✅ {out_csv}")
    best = max(shap, key=shap.__getitem__)
    print(f"🏆 Most valuable agent: {best}  (φ = {shap[best]:.6f})")


if __name__ == "__main__":
    main()
