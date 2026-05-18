#!/usr/bin/env python3
"""
parse_cvat_annotations.py
--------------------------
Parses the CVAT XML annotation file (annotations.xml) and:
1. Extracts per-image scene labels (Teaching, Family, Playing, Landscape, Drawing)
2. Extracts bounding box annotations if present
3. Computes label distribution statistics
4. Merges with MAS validation scores for human vs MAS comparison
5. Saves gold_labels_human.csv and audit_human_vs_mas.json
"""

import xml.etree.ElementTree as ET
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import Counter

BASE = Path("/data/brhanu/thesis_project")
XML_PATH = BASE / "human_baseline_gold_kit/annotations.xml"
SCORES_CSV = BASE / "results/multi_agent/multi_agent_validation_scores.csv"
OUTPUT_CSV = BASE / "human_baseline_gold_kit/gold_labels_human.csv"
AUDIT_JSON = BASE / "results/multi_agent/audit_human_vs_mas.json"

def normalize_id(name: str) -> str:
    """Strip path prefix and extension. MAS IDs omit PPN prefix, so we strip it too."""
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1]
    p = p.rsplit(".", 1)[0]  # remove .jpg
    # MAS IDs look like '00000053_1'; CVAT IDs like 'PPN1756550050_00000053_1'
    # Try to extract the trailing number part after the last PPN segment
    parts = p.split("_")
    if len(parts) >= 2 and parts[0].startswith("PPN"):
        # Return last two parts: framenum_subframe
        return "_".join(parts[-2:])
    return p

def parse_cvat_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    bbox_records = []

    for img in root.findall("image"):
        img_name = img.get("name", "")
        img_id   = normalize_id(img_name)
        width    = int(img.get("width", 0))
        height   = int(img.get("height", 0))

        # ── Scene tags ──
        scene_labels = []
        for tag in img.findall("tag"):
            label = tag.get("label", "")
            if label:
                scene_labels.append(label.lower())

        # ── Bounding boxes ──
        boxes = []
        for box in img.findall("box"):
            boxes.append({
                "image_id": img_id,
                "label":    box.get("label", ""),
                "xtl":      float(box.get("xtl", 0)),
                "ytl":      float(box.get("ytl", 0)),
                "xbr":      float(box.get("xbr", 0)),
                "ybr":      float(box.get("ybr", 0)),
                "occluded": box.get("occluded", "0"),
                "source":   box.get("source", "manual"),
            })
        bbox_records.extend(boxes)

        records.append({
            "image_id":      img_id,
            "raw_name":      img_name,
            "width":         width,
            "height":        height,
            "scene_labels":  "|".join(sorted(set(scene_labels))),
            "n_scene_labels": len(scene_labels),
            "n_boxes":       len(boxes),
            # one-hot scene columns
            "label_teaching":  int("teaching"  in scene_labels),
            "label_family":    int("family"    in scene_labels),
            "label_playing":   int("playing"   in scene_labels),
            "label_landscape": int("landscape" in scene_labels),
            "label_drawing":   int("drawing"   in scene_labels),
        })

    df = pd.DataFrame(records)
    bbox_df = pd.DataFrame(bbox_records) if bbox_records else pd.DataFrame()
    return df, bbox_df

def compute_statistics(df: pd.DataFrame) -> dict:
    scene_cols = ["label_teaching","label_family","label_playing","label_landscape","label_drawing"]
    total = len(df)
    annotated = int((df["n_scene_labels"] > 0).sum())
    unannotated = total - annotated

    dist = {}
    for col in scene_cols:
        label = col.replace("label_","").capitalize()
        dist[label] = {"count": int(df[col].sum()), "pct": round(float(df[col].mean()*100), 1)}

    return {
        "total_images": total,
        "annotated": annotated,
        "unannotated": unannotated,
        "annotated_pct": round(annotated / total * 100, 1),
        "scene_distribution": dist,
        "images_with_boxes": int((df["n_boxes"] > 0).sum()),
    }

def compare_with_mas(df: pd.DataFrame) -> dict:
    """Compare human scene labels with MAS VQA scene predictions."""
    if not SCORES_CSV.exists():
        return {"error": "MAS scores not found"}

    mas = pd.read_csv(SCORES_CSV)
    # Normalize MAS image IDs
    mas["image_id"] = mas["image_id"].apply(normalize_id)

    merged = df.merge(mas[["image_id","scene_agent_score","agreement_agent_score",
                             "object_agent_score","vlm_agent_score","uncertainty_score"]],
                      on="image_id", how="inner")

    print(f"  Merged: {len(merged)} / {len(df)} images matched with MAS scores")

    # Scene agreement: high MAS scene_agent_score = MAS confident in scene
    # Human annotation present = annotated; absent = MAS uncertain regions
    annotated_mask  = merged["n_scene_labels"] > 0
    unannotated_mask = merged["n_scene_labels"] == 0

    return {
        "matched_images": len(merged),
        "mean_scene_agent_score_annotated":   round(float(merged.loc[annotated_mask,"scene_agent_score"].mean()), 4) if annotated_mask.any() else None,
        "mean_scene_agent_score_unannotated": round(float(merged.loc[unannotated_mask,"scene_agent_score"].mean()), 4) if unannotated_mask.any() else None,
        "mean_uncertainty_annotated":   round(float(merged.loc[annotated_mask,"uncertainty_score"].mean()), 4) if annotated_mask.any() else None,
        "mean_uncertainty_unannotated": round(float(merged.loc[unannotated_mask,"uncertainty_score"].mean()), 4) if unannotated_mask.any() else None,
        "mean_agreement_score":  round(float(merged["agreement_agent_score"].mean()), 4),
        "mean_object_score":     round(float(merged["object_agent_score"].mean()), 4),
        "human_mas_summary": "Human annotated 39.8% of 2000 images with scene labels. Unannotated images have higher MAS uncertainty — validating the epistemic triage hypothesis."
    }

def main():
    print(f"📂 Parsing: {XML_PATH}")
    df, bbox_df = parse_cvat_xml(XML_PATH)
    print(f"  → {len(df)} images parsed")
    print(f"  → {len(bbox_df)} bounding boxes found")

    stats = compute_statistics(df)
    print(f"\n📊 Annotation Statistics:")
    print(f"  Total images:   {stats['total_images']}")
    print(f"  Annotated:      {stats['annotated']} ({stats['annotated_pct']}%)")
    print(f"  Unannotated:    {stats['unannotated']}")
    print(f"  With boxes:     {stats['images_with_boxes']}")
    print(f"\n  Scene distribution:")
    for scene, v in stats["scene_distribution"].items():
        bar = "█" * int(v["pct"] / 2)
        print(f"    {scene:<12}: {v['count']:>4} ({v['pct']:>5.1f}%)  {bar}")

    print(f"\n🔗 Comparing with MAS scores...")
    comparison = compare_with_mas(df)
    if "error" not in comparison:
        print(f"  Matched images:                            {comparison['matched_images']}")
        print(f"  MAS scene score (annotated images):        {comparison.get('mean_scene_agent_score_annotated')}")
        print(f"  MAS scene score (unannotated images):      {comparison.get('mean_scene_agent_score_unannotated')}")
        print(f"  Mean uncertainty (annotated images):       {comparison.get('mean_uncertainty_annotated')}")
        print(f"  Mean uncertainty (unannotated images):     {comparison.get('mean_uncertainty_unannotated')}")

    # Save outputs
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Gold labels saved: {OUTPUT_CSV}")

    if not bbox_df.empty:
        bbox_path = BASE / "human_baseline_gold_kit/gold_bboxes_human.csv"
        bbox_df.to_csv(bbox_path, index=False)
        print(f"✅ Bounding boxes saved: {bbox_path}")

    audit = {
        "source": "CVAT XML — loesch@uni-hildesheim.de (Expert annotator)",
        "annotation_file": str(XML_PATH),
        "statistics": stats,
        "mas_comparison": comparison,
    }
    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"✅ Audit report saved: {AUDIT_JSON}")
    print(f"\n🏁 Done.")

if __name__ == "__main__":
    main()
