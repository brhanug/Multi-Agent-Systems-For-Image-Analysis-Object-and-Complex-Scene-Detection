#!/usr/bin/env python3
"""Student rebuild, step 3: the training set built from the new teacher.

Keeps a GroundingDINO box when (a) its score reaches 0.35, GroundingDINO's published default box
threshold -- not tuned on any evaluation picture -- and (b) LLaVA, asked about that class, confirms it
is in the picture (P(yes) >= 0.5): the multi-teacher consensus the rebuild plan named in advance.
Pictures left without boxes stay in as background, so the student also learns when to say nothing.
Train and validation are split by book (90/10), so pages of one book never sit on both sides.

Writes data/yolo11_gdino_consensus/{images,labels}/{train,val} (images symlinked) and data.yaml with
the eight frozen classes.

    python build_gdino_student_dataset.py [--verify]
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
LABELS = BASE / "results/signal_rebuild/student_rebuild/gdino_labels"
CS = BASE / "results/signal_rebuild/class_scores"
IMG = BASE / "final_dataset/images/restored"
DS = BASE / "data/yolo11_gdino_consensus"
STATS = BASE / "results/signal_rebuild/student_rebuild/dataset_stats.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
BOX_THRESHOLD, LLAVA_YES, VAL_SHARE, SEED = 0.35, 0.5, 0.10, 20260919


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    L = m["raw"][:, ms.index("llava")]
    rows = [json.loads(l) for f in sorted(glob.glob(str(LABELS / "shard_*.jsonl"))) for l in open(f)]
    books = sorted({r["image_id"].split("/")[0] for r in rows})
    rng = np.random.default_rng(SEED)
    val_books = set(rng.choice(books, size=max(1, int(len(books) * VAL_SHARE)), replace=False))
    kept, dropped_score, dropped_llava, per_split = Counter(), 0, 0, Counter()
    lines = {}
    for r in rows:
        iid = r["image_id"]; out = []
        for c, x1, y1, x2, y2, sc in r["boxes"]:
            if sc < BOX_THRESHOLD: dropped_score += 1; continue
            if L[pos[iid], mc.index(c)] < LLAVA_YES: dropped_llava += 1; continue
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            out.append(f"{CLASSES.index(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"); kept[c] += 1
        lines[iid] = out
    stats = {"pictures": len(rows), "books": len(books), "val_books": len(val_books),
             "box_threshold": BOX_THRESHOLD, "llava_yes": LLAVA_YES,
             "boxes_kept_per_class": {c: kept[c] for c in CLASSES}, "boxes_dropped_low_score": dropped_score,
             "boxes_dropped_llava_disagrees": dropped_llava,
             "pictures_with_boxes": sum(1 for v in lines.values() if v), "background_pictures": sum(1 for v in lines.values() if not v)}
    if a.verify:
        same = json.loads(STATS.read_text()) == json.loads(json.dumps(stats)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    for split in ("train", "val"):
        (DS / "images" / split).mkdir(parents=True, exist_ok=True); (DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    for iid, out in lines.items():
        split = "val" if iid.split("/")[0] in val_books else "train"; per_split[split] += 1
        stem = iid.replace("/", "_"); link = DS / "images" / split / f"{stem}.jpg"
        if not link.exists(): link.symlink_to(IMG / f"{iid}.jpg")
        (DS / "labels" / split / f"{stem}.txt").write_text("\n".join(out) + ("\n" if out else ""))
    (DS / "data.yaml").write_text(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    stats["pictures_per_split"] = dict(per_split)
    STATS.write_text(json.dumps({k: v for k, v in stats.items() if k != "pictures_per_split"}, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
