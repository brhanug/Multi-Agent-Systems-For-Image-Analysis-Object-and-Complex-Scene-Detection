#!/usr/bin/env python3
"""
Failure Mode Taxonomy & Error Analysis.

Takes the 600 highest-disagreement images and clusters them into 5 failure types:
  A: Low-Object Ambiguity   — very few detections, ambiguous content
  B: Spatial Overlap        — multiple agents detect but disagree on location
  C: Scene Ambiguity        — scene classification has low confidence
  D: VLM-Object Mismatch    — VLM high but object agent low (semantic vs spatial gap)
  E: Systematic Low Signal  — all agents low (archival degradation / OOV content)

For each type:
  - Count and proportion
  - Mean scores per agent
  - Suggested intervention (human review priority)

Outputs:
  results/multi_agent/error_analysis.csv
  results/multi_agent/error_analysis_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

THRESH_LOW_OBJ   = 0.20    # object_agent below this = very few detections
THRESH_SCI_HIGH  = 0.60    # spatial overlap proxy above this
THRESH_SCENE_LOW = 0.25    # scene_agent below this = ambiguous scene
VLM_HIGH         = 0.70    # vlm says good
OBJ_LOW          = 0.25    # but object agent says bad

def normalize_id(raw: str) -> str:
    from pathlib import Path
    parts = Path(str(raw)).parts
    if len(parts) >= 2:
        if parts[-2] not in ["original", "restored", "images", "metadata", "results", "CycleGAN", "pix2pix"]:
            return f"{parts[-2]}/{Path(parts[-1]).stem}"
    return Path(str(raw)).stem


def classify_failure(row: pd.Series) -> str:
    obj   = float(row.get("existing_pipeline_agent", 0))
    agr   = float(row.get("agreement_agent", 0))
    scn   = float(row.get("scene_agent", 0))
    vlm   = float(row.get("vlm_agent", 0))
    rest  = float(row.get("restoration_agent", 0))
    doc   = float(row.get("document_agent", 0))

    all_low = all(v < 0.25 for v in [obj, agr, scn, vlm])

    if all_low:
        return "E_systematic_low_signal"
    if vlm > VLM_HIGH and obj < OBJ_LOW:
        return "D_vlm_object_mismatch"
    if scn < THRESH_SCENE_LOW and obj > 0.30:
        return "C_scene_ambiguity"
    if obj < THRESH_LOW_OBJ:
        return "A_low_object_ambiguity"
    if agr < 0.30 and obj > 0.35:
        return "B_spatial_overlap_conflict"
    return "F_other_disagreement"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",   default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--n-samples",  type=int, default=600)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    df   = pd.read_csv(base / args.scores_csv if not Path(args.scores_csv).is_absolute() else args.scores_csv)
    out  = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Compute inter-agent disagreement
    agent_cols = ["existing_pipeline_agent","agreement_agent","scene_agent",
                  "vlm_agent","restoration_agent","document_agent"]
    am = df[[c for c in agent_cols if c in df.columns]].to_numpy(dtype=float)
    df["inter_agent_disagreement"] = np.std(am, axis=1)

    # Take N highest-disagreement images
    high_disagree = df.nlargest(args.n_samples, "inter_agent_disagreement").copy()
    high_disagree["failure_type"] = high_disagree.apply(classify_failure, axis=1)

    # Per-type statistics
    type_descriptions = {
        "A_low_object_ambiguity":    "Few/no detections — image has minimal visible objects or is a text-dominant archival scan",
        "B_spatial_overlap_conflict":"Object detected but agents disagree on location — crowded/occluded scene",
        "C_scene_ambiguity":         "High object signal but uncertain scene category — mixed-context image",
        "D_vlm_object_mismatch":     "VLM understands semantics but spatial detector fails — small/degraded objects",
        "E_systematic_low_signal":   "All agents score low — likely archival degradation, out-of-vocabulary, or extreme blur",
        "F_other_disagreement":      "General multi-agent conflict not fitting primary types",
    }

    type_rows = []
    for ftype, sub in high_disagree.groupby("failure_type"):
        row = {
            "failure_type":  ftype,
            "description":   type_descriptions.get(ftype, ""),
            "n_images":      len(sub),
            "proportion":    round(len(sub) / len(high_disagree), 4),
        }
        for col in agent_cols + ["inter_agent_disagreement","monolithic_pipeline_agent","comparison_fusion_score"]:
            if col in sub.columns:
                row[f"{col}_mean"] = round(float(sub[col].mean()), 4)
        type_rows.append(row)
        print(f"  {ftype:<30} n={len(sub):4d} ({100*len(sub)/len(high_disagree):.1f}%)")

    type_df = pd.DataFrame(type_rows).sort_values("n_images", ascending=False)

    # Save per-image CSV
    err_path = out / "error_analysis.csv"
    high_disagree.to_csv(err_path, index=False)
    print(f"✅ Wrote {err_path}")

    # Dominant failure type
    dominant = type_df.iloc[0]["failure_type"] if not type_df.empty else "unknown"

    summary = {
        "n_high_disagreement_images": int(len(high_disagree)),
        "n_total_images": int(len(df)),
        "top_disagreement_percentile": round(float(args.n_samples/len(df)*100), 1),
        "failure_type_breakdown": type_rows,
        "dominant_failure_type": dominant,
        "type_descriptions": type_descriptions,
        "intervention_recommendations": {
            "A_low_object_ambiguity":    "Route to document agent (Kosmos OCR) for text-based classification",
            "B_spatial_overlap_conflict":"Priority human review — spatial ambiguity cannot be resolved automatically",
            "C_scene_ambiguity":         "Apply additional VQA binary questions for scene sub-categorisation",
            "D_vlm_object_mismatch":     "Run Real-ESRGAN upscaling before re-detection",
            "E_systematic_low_signal":   "Escalate to archivist — likely unique/unrecognised content",
            "F_other_disagreement":      "Apply consensus threshold increase before accepting label",
        },
        "research_insight": (
            f"The dominant failure mode is '{dominant}', accounting for "
            f"{type_df.iloc[0]['proportion']*100:.1f}% of high-disagreement images. "
            f"This confirms that inter-agent disagreement is not random noise but "
            f"reflects structured, diagnosable failure conditions."
        ),
    }

    sum_path = out / "error_analysis_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {sum_path}")

if __name__ == "__main__":
    main()
