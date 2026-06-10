#!/usr/bin/env python3
"""
Exp-E: Ground-Truth Bounding Box IoU Error Taxonomy (Priority 2).

Uses the 300-image human-annotated set (LabelStudio + YOLO labels) to produce
a proper per-class spatial error taxonomy. For each image, compares:
  - Human bounding boxes (ground truth from LabelStudio annotation)
  - YOLO pseudo-labels (from human_spatial_audit/labels/)

Categorises each human box into one of four error types against YOLO:
  1. MATCH        : best IoU >= 0.5 (YOLO agrees on location AND class)
  2. LOCALISATION : best IoU in [0.1, 0.5) AND matching class exists (rough position)
  3. MISSED       : best IoU < 0.1 across all YOLO boxes (False Negative)
  4. CLASS_ERROR  : best IoU >= 0.1 but under wrong class (classification mistake)

Per class (person, child, horse, building, etc.) computes:
  - Count and proportion of each error type
  - Mean IoU per class

Also cross-tabulates by scene type and image clarity (Faded/Blurry/Clear)
to answer RQ7 sub-question: do degraded images produce more error types?

Outputs:
  results/multi_agent/exp_e_iou_error_taxonomy.csv   (per-box rows)
  results/multi_agent/exp_e_per_class_summary.csv    (per-class aggregation)
  results/multi_agent/exp_e_error_taxonomy_summary.json
"""
from __future__ import annotations
import sqlite3, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

DB_PATH      = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")
YOLO_LABEL_DIR = Path("human_spatial_audit/labels")
OUT_BOX_CSV  = Path("results/multi_agent/exp_e_iou_error_taxonomy.csv")
OUT_CLS_CSV  = Path("results/multi_agent/exp_e_per_class_summary.csv")
OUT_JSON     = Path("results/multi_agent/exp_e_error_taxonomy_summary.json")
MAX_TASK     = 300

# YOLO 10-class mapping (index → name)
IDX_TO_CLASS = {
    0: "person", 1: "child", 2: "horse", 3: "building",
    4: "weapon", 5: "vehicle", 6: "tree", 7: "clothing",
    8: "text", 9: "animal"
}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

# ── filename normalisation ────────────────────────────────────────────────────
def normalize_fn(fn: str) -> str:
    """Strip PPN prefix: PPN1845710797_00000620_1.jpg → 00000620_1.jpg"""
    fn = fn.split("/")[-1]
    m = re.match(r"PPN[\dA-Z]+_(.+)", fn)
    return m.group(1) if m else fn

def build_yolo_lookup(label_dir: Path) -> dict:
    lookup = {}
    for f in label_dir.glob("*.txt"):
        norm = normalize_fn(f.name).replace(".txt", "")
        lookup[norm] = f
    return lookup

# ── geometry ─────────────────────────────────────────────────────────────────
def percent_to_abs(x, y, w, h):
    """LabelStudio percent → 0-100 coordinate space."""
    return x, y, x + w, y + h

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter)

# ── loaders ───────────────────────────────────────────────────────────────────
def load_annotations(max_task=300):
    """Returns list of dicts: {filename, scene, clarity, confidence, boxes:[{label, xyxy}]}"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT t.id, t.data, tc.result FROM task t
                 JOIN task_completion tc ON t.id=tc.task_id
                 WHERE t.id <= ? ORDER BY t.id""", (max_task,))
    rows = c.fetchall()
    conn.close()

    records = []
    for tid, data_str, res_str in rows:
        data   = json.loads(data_str)
        result = json.loads(res_str)
        if not isinstance(result, list) or not result:
            continue
        filename = normalize_fn(data.get("image", "").split("/")[-1])
        scene = clarity = confidence = None
        boxes = []
        for item in result:
            fn = item.get("from_name", "")
            v  = item.get("value", {})
            if fn == "scene" and v.get("choices"):
                scene = v["choices"][0].lower()
            elif fn == "clarity" and v.get("choices"):
                clarity = v["choices"][0]
            elif fn == "confidence" and v.get("choices"):
                confidence = v["choices"][0]
            elif fn == "label" and item.get("type") == "rectanglelabels":
                labels = v.get("rectanglelabels", [])
                if labels:
                    x1, y1, x2, y2 = percent_to_abs(v["x"], v["y"], v["width"], v["height"])
                    boxes.append({
                        "label": labels[0].lower(),
                        "xyxy": (x1, y1, x2, y2)
                    })
        records.append({
            "filename":   filename,
            "scene":      scene,
            "clarity":    clarity,
            "confidence": confidence,
            "boxes":      boxes
        })
    return records

def load_yolo_boxes_from_lookup(lookup, filename):
    stem = Path(filename).stem
    f = lookup.get(stem)
    if f is None or not f.exists():
        return []
    boxes = []
    with open(f) as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_idx = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * 100
            y1 = (cy - h/2) * 100
            x2 = (cx + w/2) * 100
            y2 = (cy + h/2) * 100
            boxes.append({
                "class_idx": cls_idx,
                "class_name": IDX_TO_CLASS.get(cls_idx, f"cls{cls_idx}"),
                "xyxy": (x1, y1, x2, y2)
            })
    return boxes

# ── error classification ──────────────────────────────────────────────────────
def classify_box(human_box: dict, yolo_boxes: list) -> dict:
    """
    Returns error type for one human box vs all YOLO boxes.
    
    Error types:
      MATCH        : best IoU ≥ 0.5 with same-class YOLO box
      LOCALISATION : best IoU in [0.1, 0.5) with same-class YOLO box
      CLASS_ERROR  : best overall IoU ≥ 0.1 but only with different-class YOLO box
      MISSED       : best overall IoU < 0.1 (nothing near this box)
    """
    h_label = human_box["label"]
    h_box   = human_box["xyxy"]

    if not yolo_boxes:
        return {"error_type": "MISSED", "best_iou": 0.0, "best_iou_class": None}

    # IoU with same-class YOLO boxes
    same_class = [yb for yb in yolo_boxes if yb["class_name"] == h_label]
    best_same_iou = max((iou(h_box, yb["xyxy"]) for yb in same_class), default=0.0)

    # IoU with any YOLO box (any class)
    any_iou_pairs = [(iou(h_box, yb["xyxy"]), yb["class_name"]) for yb in yolo_boxes]
    best_any_iou, best_any_class = max(any_iou_pairs, key=lambda x: x[0])

    if best_same_iou >= 0.5:
        error_type = "MATCH"
        best_iou   = best_same_iou
        best_class = h_label
    elif best_same_iou >= 0.1:
        error_type = "LOCALISATION"
        best_iou   = best_same_iou
        best_class = h_label
    elif best_any_iou >= 0.1:
        error_type = "CLASS_ERROR"
        best_iou   = best_any_iou
        best_class = best_any_class
    else:
        error_type = "MISSED"
        best_iou   = best_any_iou
        best_class = best_any_class

    return {
        "error_type":     error_type,
        "best_iou":       round(best_iou, 4),
        "best_iou_class": best_class
    }

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("📥 Loading human annotations …")
    records = load_annotations(MAX_TASK)
    print(f"   {len(records)} images with annotations.")

    print("📥 Building YOLO label lookup …")
    yolo_lookup = build_yolo_lookup(YOLO_LABEL_DIR)
    print(f"   {len(yolo_lookup)} YOLO label files indexed.")

    box_rows = []
    per_class_errors = defaultdict(lambda: defaultdict(int))   # class → error_type → count
    per_class_ious   = defaultdict(list)                        # class → [iou values]
    per_scene_errors = defaultdict(lambda: defaultdict(int))    # scene → error_type → count
    per_clarity_errors = defaultdict(lambda: defaultdict(int))  # clarity → error_type → count
    total_human_boxes = 0
    images_with_yolo  = 0

    for rec in records:
        yolo_boxes = load_yolo_boxes_from_lookup(yolo_lookup, rec["filename"])
        if yolo_boxes:
            images_with_yolo += 1

        for hbox in rec["boxes"]:
            total_human_boxes += 1
            result = classify_box(hbox, yolo_boxes)
            row = {
                "filename":   rec["filename"],
                "scene":      rec["scene"] or "unknown",
                "clarity":    rec["clarity"] or "unknown",
                "confidence": rec["confidence"] or "unknown",
                "human_label": hbox["label"],
                **result
            }
            box_rows.append(row)
            cls = hbox["label"]
            et  = result["error_type"]
            per_class_errors[cls][et] += 1
            per_class_ious[cls].append(result["best_iou"])
            if rec["scene"]:
                per_scene_errors[rec["scene"]][et] += 1
            if rec["clarity"]:
                per_clarity_errors[rec["clarity"]][et] += 1

    print(f"\n📊 Box-Level IoU Error Taxonomy (Exp-E):")
    print(f"   Total human boxes analysed: {total_human_boxes}")
    print(f"   Images with YOLO labels:    {images_with_yolo}")
    print()

    # Global error distribution
    global_errors = defaultdict(int)
    for row in box_rows:
        global_errors[row["error_type"]] += 1
    total = sum(global_errors.values())
    print("   Global error distribution:")
    for et in ["MATCH", "LOCALISATION", "CLASS_ERROR", "MISSED"]:
        n = global_errors[et]
        print(f"     {et:12s}: {n:4d} ({100*n/total:.1f}%)")

    print()
    print("   Per-class breakdown:")

    # Per-class summary
    cls_summary_rows = []
    for cls in sorted(per_class_errors.keys()):
        counts = per_class_errors[cls]
        total_cls = sum(counts.values())
        mean_iou  = float(np.mean(per_class_ious[cls])) if per_class_ious[cls] else 0.0
        row = {
            "class": cls,
            "n_human_boxes": total_cls,
            "mean_best_iou": round(mean_iou, 4),
            "MATCH":        counts.get("MATCH", 0),
            "LOCALISATION": counts.get("LOCALISATION", 0),
            "CLASS_ERROR":  counts.get("CLASS_ERROR", 0),
            "MISSED":       counts.get("MISSED", 0),
            "match_pct":        round(100 * counts.get("MATCH",0) / total_cls, 1) if total_cls else 0,
            "localisation_pct": round(100 * counts.get("LOCALISATION",0) / total_cls, 1) if total_cls else 0,
            "class_error_pct":  round(100 * counts.get("CLASS_ERROR",0) / total_cls, 1) if total_cls else 0,
            "missed_pct":       round(100 * counts.get("MISSED",0) / total_cls, 1) if total_cls else 0,
        }
        cls_summary_rows.append(row)
        print(f"     {cls:10s} n={total_cls:3d} | mIoU={mean_iou:.3f} | "
              f"MATCH={row['match_pct']:5.1f}% LOC={row['localisation_pct']:5.1f}% "
              f"CLS={row['class_error_pct']:5.1f}% MISS={row['missed_pct']:5.1f}%")

    # Per-scene error breakdown
    print()
    print("   Per-scene error type proportions:")
    scene_rows = []
    for scene in sorted(per_scene_errors.keys()):
        counts = per_scene_errors[scene]
        total_s = sum(counts.values())
        miss_pct = round(100 * counts.get("MISSED",0) / total_s, 1) if total_s else 0
        match_pct = round(100 * counts.get("MATCH",0) / total_s, 1) if total_s else 0
        print(f"     {scene:10s} n={total_s:3d} | MATCH={match_pct:5.1f}% MISS={miss_pct:5.1f}%")
        scene_rows.append({"scene": scene, "n_boxes": total_s, "match_pct": match_pct,
                            "missed_pct": miss_pct,
                            "localisation_pct": round(100*counts.get("LOCALISATION",0)/total_s,1) if total_s else 0,
                            "class_error_pct": round(100*counts.get("CLASS_ERROR",0)/total_s,1) if total_s else 0})

    # Per-clarity error breakdown
    print()
    print("   Per-clarity error type proportions:")
    clarity_rows = []
    for clar in sorted(per_clarity_errors.keys()):
        counts = per_clarity_errors[clar]
        total_c = sum(counts.values())
        miss_pct  = round(100*counts.get("MISSED",0)/total_c,1) if total_c else 0
        match_pct = round(100*counts.get("MATCH",0)/total_c,1) if total_c else 0
        print(f"     {clar:8s} n={total_c:3d} | MATCH={match_pct:5.1f}% MISS={miss_pct:5.1f}%")
        clarity_rows.append({"clarity": clar, "n_boxes": total_c, "match_pct": match_pct,
                             "missed_pct": miss_pct,
                             "localisation_pct": round(100*counts.get("LOCALISATION",0)/total_c,1) if total_c else 0,
                             "class_error_pct": round(100*counts.get("CLASS_ERROR",0)/total_c,1) if total_c else 0})

    # Write box-level CSV
    import csv
    OUT_BOX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BOX_CSV, "w", newline="") as f:
        if box_rows:
            writer = csv.DictWriter(f, fieldnames=list(box_rows[0].keys()))
            writer.writeheader()
            writer.writerows(box_rows)
    print(f"\n✅ Wrote {OUT_BOX_CSV} ({len(box_rows)} box rows)")

    # Write per-class CSV
    with open(OUT_CLS_CSV, "w", newline="") as f:
        if cls_summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(cls_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cls_summary_rows)
    print(f"✅ Wrote {OUT_CLS_CSV} ({len(cls_summary_rows)} class rows)")

    # Write JSON summary
    summary = {
        "experiment": "Exp-E: Ground-Truth Bounding Box IoU Error Taxonomy (Priority 2)",
        "n_images_annotated": len(records),
        "n_images_with_yolo_labels": images_with_yolo,
        "n_human_boxes_total": total_human_boxes,
        "global_error_distribution": {
            et: {"count": int(global_errors[et]),
                 "pct": round(100*global_errors[et]/total, 1)}
            for et in ["MATCH", "LOCALISATION", "CLASS_ERROR", "MISSED"]
        },
        "per_class_summary": cls_summary_rows,
        "per_scene_breakdown": scene_rows,
        "per_clarity_breakdown": clarity_rows,
        "key_findings": {},
        "interpretation": (
            "Four-way spatial error taxonomy on the 300-image annotated set. "
            "MISSED errors (FN) and LOCALISATION errors reveal the primary failure modes "
            "of YOLO pseudo-labels on degraded historical scans. "
            "CLASS_ERROR reveals label boundary confusion (e.g., person/child). "
            "Per-clarity breakdown tests whether image degradation (Blurry/Faded) increases MISSED rates."
        )
    }
    # Add key findings
    if cls_summary_rows:
        highest_miss = max(cls_summary_rows, key=lambda r: r["missed_pct"])
        lowest_miss  = min(cls_summary_rows, key=lambda r: r["missed_pct"])
        highest_cls_err = max(cls_summary_rows, key=lambda r: r["class_error_pct"])
        summary["key_findings"] = {
            "highest_missed_class": f"{highest_miss['class']} ({highest_miss['missed_pct']}% MISSED)",
            "lowest_missed_class":  f"{lowest_miss['class']} ({lowest_miss['missed_pct']}% MISSED)",
            "highest_class_error":  f"{highest_cls_err['class']} ({highest_cls_err['class_error_pct']}% CLASS_ERROR)",
            "clarity_comparison":   {r["clarity"]: f"MATCH={r['match_pct']}% MISS={r['missed_pct']}%" for r in clarity_rows}
        }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
