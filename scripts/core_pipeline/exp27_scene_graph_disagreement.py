#!/usr/bin/env python3
"""
exp27_scene_graph_disagreement.py
----------------------------------
Computes visual-relational mismatch scores between VLM scene graph triplets
and YOLO object detections, evaluating its utility as a predictor of model errors.

Outputs:
  results/multi_agent/exp27_scene_graph_results.json
  results/figures/exp27_scene_graph_mismatch.png
"""

import json
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
FUSION_JSON = BASE / "results/multi_agent/upgraded_agent0_fusion.json"
SG_JSONL = BASE / "final_dataset/metadata/scene_graphs_v1.jsonl"
OUT_DIR = BASE / "results/multi_agent"
FIG_DIR = BASE / "results/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Helper to normalize/clean nouns
def clean_noun(noun):
    noun = noun.strip().lower()
    # map common plurals and synonyms to singular
    noun = re.sub(r's$', '', noun)
    if noun in ["boy", "girl", "kid"]:
        return "child"
    if noun in ["man", "woman", "personne"]:
        return "person"
    if noun in ["house", "tower", "castle", "roof"]:
        return "building"
    if noun in ["tree", "flower", "shrub", "bush"]:
        return "tree"
    return noun

def main():
    print("=" * 70)
    print("EXP 27: SCENE GRAPH DISAGREEMENT TRIAGE")
    print("=" * 70)

    # 1. Load scene graphs
    print(f"📂 Loading scene graphs from: {SG_JSONL}")
    sg_dict = {}
    with open(SG_JSONL) as f:
        for line in f:
            data = json.loads(line)
            img_name = data["image"]
            # Exclude fake colorized images to evaluate native/original domain
            if "_fake_B" in img_name:
                continue
            
            # Extract triplets
            raw_sg = data.get("scene_graph", "")
            triplets = re.findall(r'\(([^)]+)\)', raw_sg)
            
            nouns = set()
            for t in triplets:
                parts = t.split(",")
                if len(parts) >= 3:
                    sub = clean_noun(parts[0])
                    obj = clean_noun(parts[2])
                    nouns.add(sub)
                    nouns.add(obj)
            
            sg_dict[img_name] = nouns

    print(f"   Loaded scene graphs for {len(sg_dict)} native images.")

    # 2. Load agent fusion metadata (which contains YOLO detections)
    print(f"📂 Loading coordinator fusion records from: {FUSION_JSON}")
    with open(FUSION_JSON) as f:
        fusion_data = json.load(f)

    # 3. Match and compute mismatch scores
    records = []
    for rec in fusion_data:
        img_name = rec.get("image_name", "")
        if img_name not in sg_dict:
            continue
            
        sg_nouns = sg_dict[img_name]
        
        # Extract YOLO detections from synthesized metadata
        detections = rec.get("synthesized_metadata", {}).get("objects", [])
        det_nouns = {clean_noun(d["name"]) for d in detections}
        
        # Calculate mismatch: fraction of scene graph nouns missing in detections
        if len(sg_nouns) == 0:
            mismatch = 0.0
        else:
            missing = sg_nouns - det_nouns
            mismatch = len(missing) / len(sg_nouns)
            
        # Error/contradiction targets
        critic = rec.get("critic_audit", {})
        contradiction = 1 if critic.get("contradictions_found") else 0
        confidence = rec.get("confidence_score", 0.5)
        
        records.append({
            "image_name": img_name,
            "sg_nouns": list(sg_nouns),
            "det_nouns": list(det_nouns),
            "mismatch_score": mismatch,
            "contradiction": contradiction,
            "confidence": confidence
        })
        
    df = pd.DataFrame(records)
    print(f"   Matched {len(df)} images with complete scene graph + YOLO data.")

    # Calculate statistics
    mean_mismatch = df["mismatch_score"].mean()
    mismatch_by_contra = df.groupby("contradiction")["mismatch_score"].mean().to_dict()
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Overall Mean Mismatch Score: {mean_mismatch:.4f}")
    for contra, score in mismatch_by_contra.items():
        label = "Contradiction Present" if contra == 1 else "Contradiction Absent"
        print(f"   ↳ {label}: {score:.4f}")

    # Correlate mismatch score with confidence/errors
    corr_matrix = df[["mismatch_score", "contradiction", "confidence"]].corr()
    print(f"\n📈 Correlation matrix:\n{corr_matrix.to_string()}")

    # Save results
    results = {
        "experiment": "Experiment 27: Scene Graph Disagreement Triage",
        "n_evaluated": len(df),
        "overall_mean_mismatch": round(float(mean_mismatch), 4),
        "mean_mismatch_by_contradiction": {str(k): round(float(v), 4) for k, v in mismatch_by_contra.items()},
        "correlation_mismatch_vs_contradiction": round(float(corr_matrix.loc["mismatch_score", "contradiction"]), 4),
        "correlation_mismatch_vs_confidence": round(float(corr_matrix.loc["mismatch_score", "confidence"]), 4),
        "image_details": df.to_dict(orient="records")
    }
    
    out_json = OUT_DIR / "exp27_scene_graph_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {out_json}")

    # Plot mismatch comparison bar chart
    plt.figure(figsize=(6, 5))
    labels = ["Contradiction Absent", "Contradiction Present"]
    values = [mismatch_by_contra.get(0, 0.0) * 100, mismatch_by_contra.get(1, 0.0) * 100]
    
    plt.bar(labels, values, color=["#1f77b4", "#d62728"], width=0.5, edgecolor="black", alpha=0.85)
    plt.title("Exp 27: Relational Mismatch vs Contradiction Presence", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Relational Mismatch Score (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)

    fig_path = FIG_DIR / "exp27_scene_graph_mismatch.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to: {fig_path}")

if __name__ == "__main__":
    main()
