import pandas as pd
import numpy as np
import math
import itertools
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GOLD_DIR = BASE / "human_baseline_gold_kit"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"

AGENTS = {
    "object":    "existing_pipeline_agent",
    "agreement": "agreement_agent",
    "scene":     "scene_agent",
    "vlm":       "vlm_agent",
}
AGENT_NAMES = list(AGENTS.keys())
N = len(AGENT_NAMES)

def clean_id(name):
    s = str(name).replace("images/", "").replace("\\", "/")
    if s.endswith(".jpg") or s.endswith(".png"):
        s = s.rsplit(".", 1)[0]
    return s.strip()

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

def main():
    file_A = GOLD_DIR / "annotator_A_worksheet.csv"
    file_B = GOLD_DIR / "annotator_B_worksheet.csv"
    
    df_A = pd.read_csv(file_A)
    df_B = pd.read_csv(file_B)
    df_merged = df_A.merge(df_B, on="Image_ID", suffixes=('_A', '_B'))
    
    df_merged["clean_id"] = df_merged["Image_ID"].apply(clean_id)
    
    df_ai = pd.read_csv(SCORES_CSV)
    df_ai["clean_id"] = df_ai["image_id"].apply(clean_id)
    df_ai = df_ai.drop_duplicates("clean_id")
    
    # Filter to 300 double-annotated images
    df_300 = df_ai[df_ai["clean_id"].isin(df_merged["clean_id"])].copy()
    print(f"Loaded {len(df_300)} matched images out of 300 double-annotated ones.")
    
    # Calculate Shapley values for SAA score
    shap_vals = shapley(df_300)
    v_grand = v(df_300, tuple(AGENT_NAMES))
    total = sum(shap_vals.values())
    
    print("\nExact Shapley Values on 300 Curation Cohort (Score Curation):")
    print(f"{'Agent':12s} | {'φ':10s} | {'%':8s}")
    print("-" * 38)
    for a, phi in sorted(shap_vals.items(), key=lambda x: -x[1]):
        print(f"{a:12s} | {phi:.6f} | {phi/total*100:.2f}%")
    print(f"Grand coalition value: {v_grand:.6f}")

    print("\nMarginal (leave-one-out):")
    for a in AGENT_NAMES:
        loo = v(df_300, tuple(x for x in AGENT_NAMES if x != a))
        print(f"  Remove {a:12s}: v = {loo:.6f}  Δ = {v_grand - loo:+.6f}")

if __name__ == "__main__":
    main()
