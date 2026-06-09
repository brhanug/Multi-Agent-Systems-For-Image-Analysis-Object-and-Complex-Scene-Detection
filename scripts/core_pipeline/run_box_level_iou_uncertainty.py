#!/usr/bin/env python3
"""
Exp-F: Box-Level Inter-Annotator IoU vs SAA Uncertainty (RQ2 — upgraded).

The current thesis validates RQ2 at IMAGE level: Pearson r=0.9213 between
human ambiguity (image-level) and SAA Coordinator Uncertainty (image-level).

This experiment upgrades it to BOUNDING-BOX level:
  For each image in the 300-image audit:
    1. Extract human bounding boxes from LabelStudio (Annotator 1 completions)
    2. Load YOLO pseudo-label boxes from human_spatial_audit/labels/
    3. Compute mean IoU between human boxes and nearest YOLO box (per image)
    4. Use (1 - mean_IoU) as a "box-level ambiguity" proxy
       (low IoU = human drew very differently from the system = ambiguous)
    5. Correlate with SAA Coordinator Uncertainty for that image

Hypothesis: images where humans drew boxes very differently from YOLO
(low IoU = high 1-IoU) should have HIGH SAA uncertainty.

High Spearman ρ between (1-IoU) and SAA_U validates that the system's
uncertainty signal is grounded in the actual spatial annotation difficulty.

Outputs:
  results/multi_agent/box_level_iou_uncertainty.csv
  results/multi_agent/box_level_iou_summary.json
"""
from __future__ import annotations
import sqlite3, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr

DB_PATH    = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")
YOLO_LABEL_DIR = Path("human_spatial_audit/labels")
GOLD_CSV   = Path("results/multi_agent/gold_simulation_subset.csv")
OUT_CSV  = Path("results/multi_agent/box_level_iou_uncertainty.csv")
OUT_JSON = Path("results/multi_agent/box_level_iou_summary.json")
MAX_TASK = 300

AGENT_COLS = [
    "existing_pipeline_agent", "agreement_agent", "scene_agent",
    "vlm_agent", "restoration_agent", "document_agent",
]

# ── geometry helpers ──────────────────────────────────────────────────────────
import re as _re
def normalize_filename(fn: str) -> str:
    """Strip PPN prefix: 'PPN1845710797_00000620_1.jpg' -> '00000620_1.jpg'.
    Handles both numeric-only PPNs and those with trailing letters (e.g. PPN177133276X)."""
    fn = fn.split("/")[-1]
    m = _re.match(r"PPN[\dA-Z]+_(.+)", fn)
    return m.group(1) if m else fn

def xywh_to_xyxy(x, y, w, h, img_w=100.0, img_h=100.0):
    """Convert percent-based LabelStudio coords to absolute xyxy."""
    x1 = x * img_w / 100.0
    y1 = y * img_h / 100.0
    x2 = (x + w) * img_w / 100.0
    y2 = (y + h) * img_h / 100.0
    return x1, y1, x2, y2

def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

def best_iou(human_box, yolo_boxes):
    """Highest IoU between one human box and any YOLO box."""
    if not yolo_boxes:
        return 0.0
    return max(iou_xyxy(human_box, yb) for yb in yolo_boxes)

def build_yolo_lookup(label_dir: Path) -> dict:
    """Build dict: normalized_stem -> Path, handling PPN-prefixed filenames."""
    lookup = {}
    for f in label_dir.glob("*.txt"):
        norm = normalize_filename(f.name).replace(".txt", "")
        lookup[norm] = f
    return lookup

# ── loaders ───────────────────────────────────────────────────────────────────
def load_human_boxes(max_task=300):
    """Return dict: filename -> list of xyxy boxes (percent coords, 100x100 space)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT t.id, t.data, tc.result
        FROM task t JOIN task_completion tc ON t.id=tc.task_id
        WHERE t.id <= ? ORDER BY t.id ASC
    """, (max_task,))
    rows = c.fetchall()
    conn.close()

    out = {}
    for tid, data_str, res_str in rows:
        data   = json.loads(data_str)
        result = json.loads(res_str)
        if not isinstance(result, list):
            continue
        filename = normalize_filename(data.get("image", "").split("/")[-1])
        boxes = []
        for item in result:
            if item.get("from_name") == "label" and item.get("type") == "rectanglelabels":
                v = item.get("value", {})
                boxes.append(xywh_to_xyxy(v["x"], v["y"], v["width"], v["height"]))
        out[filename] = boxes
    return out

def load_yolo_boxes_from_lookup(lookup: dict, filename: str):
    """Load YOLO txt label for image via pre-built lookup. Returns list of xyxy (in 100x100 space)."""
    stem  = Path(filename).stem
    label_file = lookup.get(stem)
    if label_file is None or not label_file.exists():
        return []
    boxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO format: class cx cy w h (all normalized 0-1)
            _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert to 100x100 space
            x1 = (cx - w/2) * 100
            y1 = (cy - h/2) * 100
            x2 = (cx + w/2) * 100
            y2 = (cy + h/2) * 100
            boxes.append((x1, y1, x2, y2))
    return boxes

def load_machine_scores(gold_csv: Path):
    """Returns dict: filename -> mean inter-agent disagreement (std of 6 agents across image's bboxes)."""
    if not gold_csv.exists():
        return {}
    import csv as _csv
    from collections import defaultdict
    by_img = defaultdict(list)
    with open(gold_csv) as f:
        for row in _csv.DictReader(f):
            img_id   = row["image_id"]
            filename = img_id.split("/")[-1] + ".jpg"
            scores   = [float(row[c]) for c in AGENT_COLS if c in row]
            by_img[filename].append(float(np.std(scores)) if scores else 0.0)
    return {fn: float(np.mean(vals)) for fn, vals in by_img.items()}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("📥 Loading human boxes from LabelStudio …")
    human_boxes = load_human_boxes(MAX_TASK)
    print(f"   {len(human_boxes)} images with human annotations.")

    print("📥 Loading machine (SAA) uncertainty scores …")
    machine_scores = load_machine_scores(GOLD_CSV)
    print(f"   {len(machine_scores)} inter-agent disagreement scores loaded.")

    print("📥 Building YOLO label lookup …")
    yolo_lookup = build_yolo_lookup(YOLO_LABEL_DIR)
    print(f"   {len(yolo_lookup)} YOLO label files indexed.")

    rows = []
    for filename, h_boxes in human_boxes.items():
        if not h_boxes:
            continue
        y_boxes = load_yolo_boxes_from_lookup(yolo_lookup, filename)
        # Per-image: compute mean best-IoU between each human box and nearest YOLO box
        iou_vals = [best_iou(hb, y_boxes) for hb in h_boxes]
        mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0
        box_ambiguity = 1.0 - mean_iou  # high = human drew very differently from YOLO

        saa_u = machine_scores.get(filename)
        rows.append({
            "filename":      filename,
            "n_human_boxes": len(h_boxes),
            "n_yolo_boxes":  len(y_boxes),
            "mean_best_iou": round(mean_iou, 4),
            "box_ambiguity": round(box_ambiguity, 4),  # = 1 - mean_iou
            "saa_uncertainty": round(float(saa_u), 4) if saa_u is not None else None,
        })

    # Filter to rows where we have both signals
    paired = [(r["box_ambiguity"], r["saa_uncertainty"]) for r in rows if r["saa_uncertainty"] is not None]
    print(f"   {len(paired)} images with both box-IoU and SAA uncertainty.")

    if len(paired) >= 3:
        ambig_vals  = [p[0] for p in paired]
        saa_vals    = [p[1] for p in paired]
        spearman_rho, sp_p  = spearmanr(ambig_vals, saa_vals)
        pearson_r_val, pr_p = pearsonr(ambig_vals, saa_vals)
    else:
        spearman_rho = sp_p = pearson_r_val = pr_p = float("nan")

    print(f"\n📊 Box-Level Uncertainty Alignment (RQ2):")
    print(f"   Spearman ρ (box_ambiguity vs SAA_U): {spearman_rho:.4f}  (p={sp_p:.4g})")
    print(f"   Pearson  r (box_ambiguity vs SAA_U): {pearson_r_val:.4f}  (p={pr_p:.4g})")
    print(f"   Image-level baseline (thesis):        r = 0.9213")

    # Write CSV
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"✅ Wrote {OUT_CSV}")

    summary = {
        "experiment": "Exp-F: Box-Level IoU vs SAA Uncertainty (RQ2 — upgraded)",
        "n_images_analysed": len(rows),
        "n_images_paired_both_signals": len(paired),
        "spearman_rho": round(float(spearman_rho), 4) if not (spearman_rho != spearman_rho) else None,
        "spearman_p":   round(float(sp_p), 6)         if not (sp_p != sp_p)         else None,
        "pearson_r":    round(float(pearson_r_val), 4) if not (pearson_r_val != pearson_r_val) else None,
        "pearson_p":    round(float(pr_p), 6)         if not (pr_p != pr_p)         else None,
        "image_level_baseline_r": 0.9213,
        "interpretation": (
            "Box-level alignment between (1 - mean IoU) and SAA Unified Uncertainty. "
            "A positive Spearman ρ confirms that images where humans annotate spatial boxes "
            "differently from YOLO pseudo-labels are precisely the images the SAA system flags "
            "as high-uncertainty — validating RQ2 at sub-image granularity."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
