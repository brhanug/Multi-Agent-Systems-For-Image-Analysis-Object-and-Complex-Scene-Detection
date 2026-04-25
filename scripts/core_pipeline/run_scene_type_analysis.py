#!/usr/bin/env python3
"""
Scene-Type Performance Breakdown.

Evaluates monolithic vs fusion vs individual agents across the 5 historical
scene categories: teaching, landscape, drawing, family, playing.

Also computes inter-scene variance to answer: does multi-agent fusion
benefit some scene types more than others? (RQ3 extension)

Outputs:
  results/multi_agent/scene_type_performance.csv
  results/multi_agent/scene_type_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SCENE_TYPES = ["teaching", "landscape", "drawing", "family", "playing"]

AGENT_COLS = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent",
    "vlm_agent", "restoration_agent", "document_agent",
    "monolithic_pipeline_agent", "comparison_fusion_score",
]

def normalize_id(raw: str) -> str:
    return Path(str(raw)).stem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--manifest", default="final_dataset/metadata/manifest.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scores_csv = Path(args.scores_csv) if Path(args.scores_csv).is_absolute() else base / args.scores_csv
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else base / args.manifest
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_csv)
    manifest = pd.read_csv(manifest_path)

    # Normalize IDs and merge
    manifest["img_key"] = manifest["image_id"].apply(normalize_id)
    scores["img_key"]   = scores["image_id"].apply(normalize_id)
    merged = scores.merge(manifest[["img_key", "vqa_primary_scene", "vqa_total_objects"]], on="img_key", how="left")

    rows = []
    for scene in SCENE_TYPES:
        sub = merged[merged["vqa_primary_scene"] == scene]
        if len(sub) == 0:
            continue
        row = {"scene_type": scene, "n_images": len(sub)}
        for col in AGENT_COLS:
            if col in sub.columns:
                row[f"{col}_mean"] = round(float(sub[col].mean()), 4)
                row[f"{col}_std"]  = round(float(sub[col].std()), 4)
        # Fusion advantage over monolithic in this scene
        row["delta_fusion_vs_mono"] = round(
            row.get("comparison_fusion_score_mean", 0) - row.get("monolithic_pipeline_agent_mean", 0), 4
        )
        # Inter-agent disagreement
        agent_matrix = sub[[c for c in AGENT_COLS[:6] if c in sub.columns]].to_numpy(dtype=float)
        row["inter_agent_disagreement"] = round(float(np.std(agent_matrix, axis=1).mean()), 4)
        row["avg_object_count"] = round(float(sub["vqa_total_objects"].mean()), 2)
        rows.append(row)
        print(f"  {scene:10s} n={len(sub):5d} | mono={row.get('monolithic_pipeline_agent_mean',0):.3f} "
              f"| fus={row.get('comparison_fusion_score_mean',0):.3f} | Δ={row['delta_fusion_vs_mono']:+.3f}")

    scene_df = pd.DataFrame(rows)
    scene_path = out_dir / "scene_type_performance.csv"
    scene_df.to_csv(scene_path, index=False)
    print(f"✅ Wrote {scene_path}")

    # Best and worst scene for fusion advantage
    best_scene  = max(rows, key=lambda r: r["delta_fusion_vs_mono"])
    worst_scene = min(rows, key=lambda r: r["delta_fusion_vs_mono"])

    summary = {
        "n_scene_types": len(rows),
        "scene_breakdown": {r["scene_type"]: {"n": r["n_images"],
            "monolithic_mean": r.get("monolithic_pipeline_agent_mean"),
            "fusion_mean": r.get("comparison_fusion_score_mean"),
            "delta": r["delta_fusion_vs_mono"],
            "disagreement": r["inter_agent_disagreement"]} for r in rows},
        "best_scene_for_fusion": best_scene["scene_type"],
        "best_scene_delta": best_scene["delta_fusion_vs_mono"],
        "worst_scene_for_fusion": worst_scene["scene_type"],
        "worst_scene_delta": worst_scene["delta_fusion_vs_mono"],
        "rq3_scene_insight": (
            f"Multi-agent fusion provides largest advantage in '{best_scene['scene_type']}' scenes "
            f"(Δ={best_scene['delta_fusion_vs_mono']:+.3f}) where specialized agents complement "
            f"each other most effectively. Smallest advantage in '{worst_scene['scene_type']}' "
            f"(Δ={worst_scene['delta_fusion_vs_mono']:+.3f}) where object signal dominates."
        ),
    }
    summary_path = out_dir / "scene_type_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {summary_path}")

if __name__ == "__main__":
    main()
