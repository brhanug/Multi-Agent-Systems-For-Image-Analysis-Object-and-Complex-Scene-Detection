#!/usr/bin/env python3
"""Reproduces the deployed student's 0.146 mAP50 against the 300 human-boxed pictures (Chapter 3).

This is the Ultralytics validator (model.val), the same evaluator that produced the 0.768 against the
teacher's labels during training, run with the label scheme of scripts/core_pipeline/evaluate_spatial_audit.py.
It scores eight classes (the ones with human boxes: person, child, building, weapon, vehicle, tree, text,
animal). The thesis's own evaluator (run_teacher_candidates_on_300.py; all-point AP, boxes matched in order of
confidence, horse merged into animal) scores the same detections over the same eight classes at 0.123; it is
the one used for every detector compared in Section 7.3. Added 23 Nov 2026 to reconcile the two figures.

    python run_ultralytics_val_deployed.py [--device <gpu>] [--verify]   (Ultralytics sets the visible GPU itself)
"""
import argparse, json, tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE = Path("/data/brhanu/thesis_project")
AUDIT = BASE / "human_spatial_audit"
WEIGHTS = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
OUT = BASE / "results/signal_rebuild/student_rebuild/ultralytics_val_deployed_300.json"
NAMES = {0: 'person', 1: 'child', 2: 'horse', 3: 'building', 4: 'weapon', 5: 'vehicle', 6: 'tree', 7: 'clothing', 8: 'text', 9: 'animal'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); ap.add_argument("--device", default="0"); a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        y = Path(tmp) / "audit.yaml"
        yaml.safe_dump({"path": str(AUDIT), "train": "images", "val": "images", "names": NAMES}, open(y, "w"))
        r = YOLO(str(WEIGHTS)).val(data=str(y), split="val", batch=16, device=a.device, project=tmp, name="val", plots=False, verbose=False)
    out = {"evaluator": "ultralytics model.val (101-point interpolated AP at IoU 0.5)", "pictures": 300,
           "map50": round(float(r.box.map50), 4), "classes_scored": [NAMES[int(c)] for c in r.box.ap_class_index],
           "ap50": {NAMES[int(c)]: round(float(v), 4) for c, v in zip(r.box.ap_class_index, r.box.ap50)}}
    if a.verify:
        same = json.loads(OUT.read_text()) == out; print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
