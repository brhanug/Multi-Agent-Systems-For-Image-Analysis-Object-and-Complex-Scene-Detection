#!/usr/bin/env python3
"""B3, second half: when the student detector misses, is it the teacher's fault or the training's?

The student (the deployed YOLO11) was trained on the teacher's pseudo-labels. A miss has two
different causes with different fixes:
  teacher failure   the object was never in the pseudo-label, so the student could not learn it
                    -> fix the teacher or its filter;
  training failure  the pseudo-label had it and the student still does not predict it
                    -> fix the training (balance, epochs, class weights).

A. On the 300 hand-annotated pictures that also carry a teacher label (the student's own training
   and validation pictures), for every human-drawn class: did the teacher have it, did the student
   find it (a box of that class at the deployed confidence 0.25)?
B. Across the archive: of the pictures where the student finds nothing, how many had teacher boxes
   (a training failure), and how many had none although LLaVA, asked per class, sees one of the
   eight classes (P(yes) >= 0.5)?

Class ids follow the detector's training order; horse merges into animal and clothing is left out
(the frozen eight-class list; the hand check found the clothing id marks children).

    python run_teacher_vs_student.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/b3/teacher_vs_student.json"
TRAIN_ORDER = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
TO8 = {"person": "person", "child": "child", "horse": "animal", "animal": "animal", "building": "building",
       "weapon": "weapon", "vehicle": "vehicle", "tree": "tree", "text": "text"}
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]


def teacher_labels():
    out = {}
    for split in ("train", "val"):
        for f in glob.glob(str(BASE / f"data/yolo11_v2_augmented/labels/{split}/*.txt")):
            stem = Path(f).stem
            if stem.startswith("aug_"): continue
            ids = {int(float(l.split()[0])) for l in open(f) if l.strip()}
            out[stem.replace("_", "/", 1)] = {TO8[TRAIN_ORDER[i]] for i in ids if TRAIN_ORDER[i] in TO8}
    return out


def student_labels():
    out = {}
    for line in open(CS / "yolo_class_scores.jsonl"):
        r = json.loads(line)
        out[r["image_id"]] = {TO8[c] for c, n in r["count_at_0.25"].items() if n > 0 and c in TO8}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    teach, stud = teacher_labels(), student_labels()
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    both = [i for i in truth.index if i in teach]
    per_class = []
    for c in CLASSES:
        pos = [i for i in both if truth.loc[i, c] == 1]; neg = [i for i in both if truth.loc[i, c] == 0]
        t = np.array([c in teach[i] for i in pos]); s = np.array([c in stud[i] for i in pos])
        tn = np.array([c in teach[i] for i in neg]); sn = np.array([c in stud[i] for i in neg])
        per_class.append({"class": c, "human_positives_with_teacher_label": len(pos),
                          "teacher_missed": int((~t).sum()) if len(pos) else 0,
                          "teacher_had_student_missed": int((t & ~s).sum()) if len(pos) else 0,
                          "both_found": int((t & s).sum()) if len(pos) else 0,
                          "student_found_without_teacher": int((~t & s).sum()) if len(pos) else 0,
                          "teacher_false_positives": int(tn.sum()), "student_false_positives": int(sn.sum())})
    misses_teacher = sum(r["teacher_missed"] for r in per_class)
    misses_training = sum(r["teacher_had_student_missed"] for r in per_class)

    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    L = m["raw"][:, ms.index("llava")][:, [mc.index(c) for c in CLASSES]]
    llava_sees = dict(zip(ids, (L >= 0.5).any(axis=1)))
    zero = [i for i in ids if not stud.get(i)]
    in_teacher = [i for i in zero if i in teach]
    corpus = {"pictures": len(ids), "student_finds_nothing": len(zero),
              "of_those_in_student_training_data": len(in_teacher),
              "teacher_had_boxes_student_learned_nothing": int(sum(bool(teach[i]) for i in in_teacher)),
              "teacher_had_no_boxes_but_llava_sees_a_class": int(sum((not teach[i]) and llava_sees[i] for i in in_teacher)),
              "not_in_training_data_llava_sees_a_class": int(sum((i not in teach) and llava_sees[i] for i in zero)),
              "llava_sees_nothing_either": int(sum(not llava_sees[i] for i in zero))}
    out = {"experiment": "Teacher failure or training failure (B3, second half)",
           "pictures_of_the_300_with_teacher_labels": len(both),
           "misses_on_the_300": {"teacher_failure": misses_teacher, "training_failure": misses_training},
           "per_class_on_the_300": per_class, "archive": corpus}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    print(f"300 pictures with teacher labels: {len(both)} | misses: teacher failure {misses_teacher}, training failure {misses_training}")
    for r in per_class:
        print(f"  {r['class']:9s} human+ {r['human_positives_with_teacher_label']:3d} | teacher missed {r['teacher_missed']:3d} | teacher had, student missed "
              f"{r['teacher_had_student_missed']:3d} | both {r['both_found']:3d} | student alone {r['student_found_without_teacher']:2d} "
              f"| false pos: teacher {r['teacher_false_positives']}, student {r['student_false_positives']}")
    print("archive:", corpus)


if __name__ == "__main__":
    raise SystemExit(main())
