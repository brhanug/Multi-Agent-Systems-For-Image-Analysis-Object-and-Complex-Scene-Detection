#!/usr/bin/env python3
"""
Synthetic Human Baseline Simulation.

Since manual annotation of 500 images is pending, this script:
1. Uses the gold simulation subset (strong agent consensus) as a proxy for
   "images a human would agree are good/present"
2. Generates synthetic binary labels for all 10 object classes using
   VQA binary question scores, scene tags, and student detection counts
3. Evaluates pipeline vs synthetic human on class-wise precision/recall
4. Produces a proper annotation_completeness + agreement report

The key academic framing: this is NOT ground truth — it is a
"conservative synthetic oracle" baseline derived from model consensus,
suitable for measuring alignment between the pipeline and a
hypothetical expert annotator.

Outputs:
  results/multi_agent/synthetic_human_baseline.csv
  results/multi_agent/synthetic_human_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ["Person", "Child", "Horse", "Building", "Weapon",
           "Vehicle", "Tree", "Clothing", "Text", "Animal"]

# Scene → likely present classes (from domain knowledge of Colibri archive)
SCENE_CLASS_PRIORS: dict[str, list[str]] = {
    "teaching":  ["Person", "Child", "Clothing", "Building"],
    "family":    ["Person", "Child", "Clothing", "Animal"],
    "playing":   ["Person", "Child", "Animal", "Tree"],
    "landscape": ["Tree", "Animal", "Horse", "Building"],
    "drawing":   ["Person", "Clothing", "Text"],
}

def normalize_id(raw: str) -> str:
    return Path(str(raw)).stem

def infer_class_label(row: pd.Series, cls: str) -> int:
    """Infer binary class presence from available agent signals."""
    scene  = str(row.get("vqa_primary_scene", "")).lower()
    obj_n  = float(row.get("vqa_total_objects", 0))
    tags   = str(row.get("scene_tags_clip", "")).lower()
    caption= str(row.get("blip2_caption", "")).lower()
    student= float(row.get("student_v3_count", 0))
    agree  = float(row.get("agreement_agent", 0))
    vlm    = float(row.get("vlm_agent", 0))

    cls_lower = cls.lower()

    # Text in tags or caption
    tag_hit     = cls_lower in tags or cls_lower in caption
    # Prior from scene type
    scene_prior = cls in SCENE_CLASS_PRIORS.get(scene, [])
    # Object count heuristic
    obj_hit     = obj_n >= 1 and cls in ["Person", "Child", "Animal", "Horse"]
    # VLM confidence threshold (VLM "knows" about the image content)
    vlm_hit     = vlm > 0.65 and scene_prior

    # Synthesise: positive if 2+ signals agree
    signals = [tag_hit, scene_prior, obj_hit and cls in ["Person","Child","Animal","Horse"], vlm_hit]
    return int(sum(signals) >= 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",   default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--manifest",   default="final_dataset/metadata/manifest.csv")
    parser.add_argument("--worksheet",  default="human_baseline_gold_kit/labeling_worksheet.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base     = Path(args.base_dir).resolve()
    scores   = pd.read_csv(base / args.scores_csv   if not Path(args.scores_csv).is_absolute()  else args.scores_csv)
    manifest = pd.read_csv(base / args.manifest      if not Path(args.manifest).is_absolute()    else args.manifest)
    worksheet= pd.read_csv(base / args.worksheet     if not Path(args.worksheet).is_absolute()   else args.worksheet)
    out      = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Merge all signals
    manifest["img_key"] = manifest["image_id"].apply(normalize_id)
    scores["img_key"]   = scores["image_id"].apply(normalize_id)
    df = scores.merge(manifest, on="img_key", how="inner", suffixes=("","_m"))

    # Focus on the 500 worksheet images
    worksheet["img_key"] = worksheet["Image_ID"].apply(normalize_id)
    subset = df[df["img_key"].isin(worksheet["img_key"])].copy()
    if len(subset) == 0:
        # fallback: use all images if worksheet IDs don't match exactly
        print("⚠  Worksheet images not found in scores — using full dataset sample")
        subset = df.sample(min(500, len(df)), random_state=42).copy()

    print(f"📋 Subset size: {len(subset)} images")

    # Generate synthetic binary labels for each class
    for cls in CLASSES:
        subset[f"synthetic_{cls}"] = subset.apply(lambda r: infer_class_label(r, cls), axis=1)

    # Pipeline detection labels (from student_v3_count and agent scores)
    for cls in CLASSES:
        # Pipeline "detects" class if object count > 0 and agreement above threshold
        subset[f"pipeline_{cls}"] = ((subset["existing_pipeline_agent"] > 0.3) &
                                      (subset["student_v3_count"] > 0 if "student_v3_count" in subset.columns else True)).astype(int)

    # Per-class metrics: pipeline vs synthetic human
    class_rows = []
    for cls in CLASSES:
        syn_col  = f"synthetic_{cls}"
        pipe_col = f"pipeline_{cls}"
        tp = int(((subset[syn_col]==1) & (subset[pipe_col]==1)).sum())
        fp = int(((subset[syn_col]==0) & (subset[pipe_col]==1)).sum())
        fn = int(((subset[syn_col]==1) & (subset[pipe_col]==0)).sum())
        tn = int(((subset[syn_col]==0) & (subset[pipe_col]==0)).sum())
        n_pos = int(subset[syn_col].sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        agree_rate = (tp+tn)/len(subset)
        class_rows.append({
            "class": cls,
            "n_synthetic_positive": n_pos,
            "n_synthetic_positive_pct": round(100*n_pos/len(subset),1),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "pipeline_human_agreement": round(agree_rate, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        print(f"  {cls:<12} pos={n_pos:4d} ({100*n_pos/len(subset):.0f}%)  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  agree={agree_rate:.3f}")

    class_df = pd.DataFrame(class_rows)

    # Overall macro metrics
    macro_p  = float(class_df["precision"].mean())
    macro_r  = float(class_df["recall"].mean())
    macro_f1 = float(class_df["f1"].mean())
    macro_ag = float(class_df["pipeline_human_agreement"].mean())

    print(f"\n  Macro avg: P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  agree={macro_ag:.3f}")

    # Save per-image file with synthetic labels
    cols_to_save = (["image_id", "img_key", "comparison_fusion_score", "monolithic_pipeline_agent",
                     "agreement_agent", "vlm_agent", "vqa_primary_scene"] +
                    [f"synthetic_{c}" for c in CLASSES] + [f"pipeline_{c}" for c in CLASSES])
    cols_to_save = [c for c in cols_to_save if c in subset.columns]
    subset[cols_to_save].to_csv(out / "synthetic_human_baseline.csv", index=False)
    print(f"✅ Wrote {out / 'synthetic_human_baseline.csv'}")

    summary = {
        "n_images": int(len(subset)),
        "method": "Synthetic oracle: 2-of-4 signal consensus (caption, scene prior, object count, VLM)",
        "provenance": "proxy (no manual annotation — synthetic labels only)",
        "macro_precision":  round(macro_p, 4),
        "macro_recall":     round(macro_r, 4),
        "macro_f1":         round(macro_f1, 4),
        "macro_agreement":  round(macro_ag, 4),
        "per_class": class_rows,
        "top_class_by_f1":  class_df.loc[class_df["f1"].idxmax(), "class"],
        "weakest_class_by_f1": class_df.loc[class_df["f1"].idxmin(), "class"],
        "academic_note": (
            "This is a conservative synthetic oracle baseline, NOT a human ground truth. "
            "It uses 2-of-4 signal consensus to approximate what a human annotator "
            "would label based on scene context, caption content, VQA scores, and object detection. "
            "Manual validation of these labels is the recommended next step."
        ),
    }
    with open(out / "synthetic_human_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {out / 'synthetic_human_summary.json'}")

if __name__ == "__main__":
    main()
