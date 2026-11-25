#!/usr/bin/env python3
"""What the other systems say about the teacher's pseudo-labels (Phase 0, B2 -- first part).

Two pseudo-label sets exist and neither records what its class ids meant when written:
  final_dataset   final_dataset/labels/yolo_labels (4,879 files, 3,622 with boxes), rewritten
                  on 19 Feb by a remap that assumed the detector's 10-class order;
  v1_refresh      data/yolo11_v1_refresh (8,212 pictures), the set the detector trained on.

For every class id, this compares the pictures carrying that id with the other pictures of
the same set, using the per-class score matrix (within-corpus percentiles, so systems are
comparable). For each system and class it reports the lift: mean percentile with the id
minus mean percentile without. The class whose score rises most is what the other systems
see where the teacher put that id -- evidence about the id's meaning, and about whether the
teacher and the rest of the ensemble agree. It does not replace the hand check (Task F).

    python run_teacher_label_agreement.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
M = BASE / "results/signal_rebuild/class_scores/class_score_matrix.npz"
OUT = BASE / "results/signal_rebuild/class_scores/teacher_label_agreement.json"
SETS = {"final_dataset": [BASE / "final_dataset/labels/yolo_labels"],
        "v1_refresh": [BASE / f"data/yolo11_v1_refresh/labels/{s}" for s in ("train", "val", "test")]}
NAMES = {"final_dataset_as_remapped": ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"],
         "detector_training_order": ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"],
         "dataset_yaml_now": ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text", "hat", "furniture"]}


def norm(stem):
    return stem.replace("_", "/", 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(M, allow_pickle=True)
    ids = list(m["ids"]); pos = {i: k for k, i in enumerate(ids)}
    classes, systems, P = list(m["classes"]), list(m["systems"]), m["percentile"].astype(np.float64)
    out = {"experiment": "Other systems on the teacher's pseudo-labels (B2, part 1)", "name_lists_on_record": NAMES, "sets": {}}
    for name, dirs in SETS.items():
        files = [f for d in dirs for f in glob.glob(str(d / "*.txt"))]
        present = {}
        for f in files:
            k = pos.get(norm(Path(f).stem))
            if k is None: continue
            present[k] = {int(float(l.split()[0])) for l in open(f) if l.strip()}
        rows = np.array(sorted(present))
        per_id = []
        for cid in sorted({c for s in present.values() for c in s}):
            has = np.array([cid in present[r] for r in rows])
            entry = {"class_id": cid, "pictures_with_id": int(has.sum()), "lift_by_system": {}}
            for si, s in enumerate(systems):
                lift = {c: round(float(np.nanmean(P[rows[has], si, ci]) - np.nanmean(P[rows[~has], si, ci])), 4)
                        for ci, c in enumerate(classes) if not np.isnan(P[rows, si, ci]).all()}
                entry["lift_by_system"][s] = dict(sorted(lift.items(), key=lambda kv: -kv[1])[:3])
            per_id.append(entry)
        out["sets"][name] = {"label_files_matched": len(present), "pictures_with_boxes": int(sum(bool(s) for s in present.values())), "per_id": per_id}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, s in out["sets"].items():
        print(f"\n{name}: {s['label_files_matched']} files, {s['pictures_with_boxes']} with boxes")
        for e in s["per_id"]:
            tops = "  ".join(f"{sy}: {', '.join(f'{c} {v:+.2f}' for c, v in d.items())}" for sy, d in e["lift_by_system"].items())
            print(f"  id {e['class_id']} ({e['pictures_with_id']} pictures)  {tops}")


if __name__ == "__main__":
    raise SystemExit(main())
