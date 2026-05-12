#!/usr/bin/env python3
"""
active_agent_adjudicator.py
----------------------------
Phase 5 advanced agentic behavior: resolves inter-agent conflicts by
querying LLaVA-OneVision as an adjudicator.

How it works:
  1. Read agent_comparison_scores.csv to find images where agents strongly
     disagree (high uncertainty = std dev across validation agents).
  2. For each disputed image, construct a structured prompt describing the
     disagreement and ask LLaVA to adjudicate.
  3. Write adjudication results to:
       results/multi_agent/adjudication_results.csv
       results/multi_agent/adjudication_summary.json

Conflict example:
    YOLO (object agent): "horse" detected with conf 0.82
    VLM agent: content classified as 0 (no objects of interest)
    → Adjudicator query: "An object detector found a horse but the VLM agent
      disagrees. Looking at this image, is there a horse present? Yes/No."

Usage:
    python active_agent_adjudicator.py --top-n 50 [--dry-run]

--dry-run: skip LLaVA query, use random adjudication for testing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

BASE       = Path(__file__).resolve().parents[2]
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
MA_CSV     = BASE / "results" / "multi_agent" / "multi_agent_validation_scores.csv"
IMAGES_DIR = BASE / "final_dataset_v1_refresh" / "images"
OUTPUT_DIR = BASE / "results" / "multi_agent"

VALIDATION_COLS = [
    "existing_pipeline_agent",  # object
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "social_composition_score", # demographic
    "geospatial_score",         # geospatial
]

DISAGREEMENT_THRESHOLD = 0.25  # std dev across validation agents to flag conflict

# ---------------------------------------------------------------------------


def find_conflicts(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return the top-n images with highest inter-agent disagreement."""
    valid = df[[c for c in VALIDATION_COLS if c in df.columns]].copy()
    
    # Priority A: Unified Uncertainty Fusion Score
    # U = alpha * sigma_agents + beta * (1 - c_bar)
    sigma_agents = valid.std(axis=1).fillna(0)
    c_bar = valid.mean(axis=1).fillna(0)
    df["conflict_score"] = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    
    conflicts = df[df["conflict_score"] >= DISAGREEMENT_THRESHOLD].copy()
    conflicts = conflicts.sort_values("conflict_score", ascending=False).head(top_n)
    print(f"⚠️  Found {len(df[df['conflict_score'] >= DISAGREEMENT_THRESHOLD])} conflicted images "
          f"(threshold={DISAGREEMENT_THRESHOLD}), processing top {len(conflicts)}")
    return conflicts


def build_prompt(row: pd.Series) -> str:
    """Create a structured adjudication prompt for one image."""
    object_v = float(row.get("existing_pipeline_agent", 0.5))
    vlm_v    = float(row.get("vlm_agent", 0.5))
    scene_v  = float(row.get("scene_agent", 0.5))
    agr_v    = float(row.get("agreement_agent", 0.5))

    dominant = "object detector" if object_v > vlm_v else "VLM"
    minority  = "VLM" if dominant == "object detector" else "object detector"
    dominant_v = max(object_v, vlm_v)
    minority_v = min(object_v, vlm_v)

    return (
        f"You are an expert historian reviewing a historical archival illustration.\n"
        f"An automated multi-agent system has produced conflicting assessments:\n"
        f"  - Object detector confidence: {object_v:.2f}\n"
        f"  - VLM content score:          {vlm_v:.2f}\n"
        f"  - Scene classifier score:     {scene_v:.2f}\n"
        f"  - Agreement metric score:     {agr_v:.2f}\n"
        f"  - Demographic profile score:  {float(row.get('social_composition_score', 0.5)):.2f}\n"
        f"  - Geospatial analyst score:   {float(row.get('geospatial_score', 0.5)):.2f}\n\n"
        f"The {dominant} (score {dominant_v:.2f}) strongly disagrees with the "
        f"{minority} (score {minority_v:.2f}).\n\n"
        f"Based on the image content, which agent is more likely correct?\n"
        f"Reply with exactly one word: either 'object' or 'vlm'.\n"
        f"Then add one short sentence (max 15 words) explaining your decision."
    )


def query_llava(prompt: str, image_path: Path | None) -> str:
    """
    Query a locally running LLaVA-OneVision server.
    Falls back to stub if server unavailable.
    """
    try:
        import requests  # type: ignore
        payload = {
            "model": "llava-onevision",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
        }
        if image_path and image_path.exists():
            import base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            payload["messages"][0]["content"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        r = requests.post("http://localhost:8000/v1/chat/completions",
                         json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[LLaVA unavailable: {exc}]"


def adjudicate_dry_run(row: pd.Series) -> str:
    """Return a random adjudication (for --dry-run testing)."""
    obj_v = float(row.get("existing_pipeline_agent", 0.5))
    vlm_v = float(row.get("vlm_agent", 0.5))
    winner = "object" if obj_v > vlm_v else "vlm"
    return f"{winner} [DRY-RUN synthetic adjudication]"


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n",   type=int, default=50,
                        help="Number of conflict images to adjudicate (default=50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLaVA; produce synthetic adjudications for testing")
    args = parser.parse_args()

    if not SCORES_CSV.exists():
        print(f"❌ Not found: {SCORES_CSV}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(SCORES_CSV)
    df["_key"] = df["image_id"].astype(str)
    df = df.drop_duplicates("_key")
    
    # Merge new Agent 4 and Agent 5 scores
    demo_csv = OUTPUT_DIR / "demographic_profile.csv"
    geo_csv  = OUTPUT_DIR / "geospatial_analysis.csv"
    if demo_csv.exists():
        demo_df = pd.read_csv(demo_csv)
        demo_df["_key"] = demo_df["image_id"].astype(str)
        df = df.merge(demo_df[["_key", "social_composition_score"]], on="_key", how="left")
        df["social_composition_score"] = df["social_composition_score"].fillna(0.5)
    if geo_csv.exists():
        geo_df = pd.read_csv(geo_csv)
        geo_df["_key"] = geo_df["image_id"].astype(str)
        df = df.merge(geo_df[["_key", "geospatial_score"]], on="_key", how="left")
        df["geospatial_score"] = df["geospatial_score"].fillna(0.5)

    conflicts = find_conflicts(df, args.top_n)

    results: list[dict] = []
    for _, row in conflicts.iterrows():
        img_id  = str(row["_key"])
        prompt  = build_prompt(row)
        # Try to locate the image
        img_path = None
        for suffix in [".jpg", ".png", ".jpeg"]:
            candidate = IMAGES_DIR / (img_id + suffix)
            if candidate.exists():
                img_path = candidate
                break

        if args.dry_run:
            response = adjudicate_dry_run(row)
        else:
            response = query_llava(prompt, img_path)

        winner = "object" if response.lower().startswith("object") else \
                 "vlm"    if response.lower().startswith("vlm")    else "unclear"

        results.append({
            "image_id":         img_id,
            "conflict_score":   round(float(row["conflict_score"]), 4),
            "object_agent":     round(float(row.get("existing_pipeline_agent", 0)), 4),
            "vlm_agent":        round(float(row.get("vlm_agent", 0)), 4),
            "scene_agent":      round(float(row.get("scene_agent", 0)), 4),
            "agreement_agent":  round(float(row.get("agreement_agent", 0)), 4),
            "demographic_agent":round(float(row.get("social_composition_score", 0)), 4),
            "geospatial_agent": round(float(row.get("geospatial_score", 0)), 4),
            "adjudication_raw": response,
            "adjudication_winner": winner,
            "image_found":      int(img_path is not None),
        })
        status = "✓" if winner != "unclear" else "?"
        print(f"  [{status}] {img_id}: conflict={row['conflict_score']:.3f} → {winner}")

    # Write outputs
    result_df = pd.DataFrame(results)
    out_csv  = OUTPUT_DIR / "adjudication_results.csv"
    out_json = OUTPUT_DIR / "adjudication_summary.json"
    result_df.to_csv(out_csv, index=False)

    summary = {
        "total_adjudicated":  len(results),
        "dry_run":            args.dry_run,
        "winner_counts":      result_df["adjudication_winner"].value_counts().to_dict(),
        "mean_conflict_score": round(float(result_df["conflict_score"].mean()), 4),
        "object_wins_pct":    round(float((result_df["adjudication_winner"] == "object").mean()) * 100, 1),
        "vlm_wins_pct":       round(float((result_df["adjudication_winner"] == "vlm").mean()) * 100, 1),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ {out_csv}")
    print(f"✅ {out_json}")
    print(f"\n📊 Object wins: {summary['object_wins_pct']}%  |  "
          f"VLM wins: {summary['vlm_wins_pct']}%  |  "
          f"Mean conflict: {summary['mean_conflict_score']:.3f}")
    if args.dry_run:
        print("\n⚠️  DRY-RUN mode — all adjudications are synthetic. "
              "Run without --dry-run and with a LLaVA server on port 8000 for real results.")


if __name__ == "__main__":
    main()
