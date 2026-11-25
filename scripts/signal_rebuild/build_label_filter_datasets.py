#!/usr/bin/env python3
"""Label-filter experiment, step 1: three training sets that differ only in which teacher boxes are kept.

Does model agreement work as a filter on training labels? The student rebuild kept a GroundingDINO
box only when LLaVA confirmed its class in the picture, and the new student scored 0.205 against
0.123 for the old one -- but that run changed the teacher and the filter at once. Here the teacher is
fixed (the same GroundingDINO boxes at score >= 0.35) and only the filter varies:

  all        every box at score >= 0.35, no filter
  consensus  only boxes whose class LLaVA confirms (P(yes) >= 0.5) -- the rebuilt student's set
  random     from "all", the same number of boxes removed per class as the consensus filter
             removes, but chosen at random: the control for "fewer labels" against "better labels"

Same pictures, same book split (90/10, the rebuild's seed), same background pictures, so only the
kept boxes differ. Writes data/label_filter/<variant>/{images,labels}/{train,val} (images symlinked)
and data.yaml, and the box counts to results/signal_rebuild/label_filter/datasets.json.

    python build_label_filter_datasets.py [--verify]
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
LABELS = BASE / "results/signal_rebuild/student_rebuild/gdino_labels"
CS = BASE / "results/signal_rebuild/class_scores"
IMG = BASE / "final_dataset/images/restored"
DS = BASE / "data/label_filter"
OUT = BASE / "results/signal_rebuild/label_filter/datasets.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
BOX_THRESHOLD, LLAVA_YES, VAL_SHARE, SPLIT_SEED, DROP_SEED = 0.35, 0.5, 0.10, 20260919, 20261007
VARIANTS = ["all", "consensus", "random"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    L = m["raw"][:, ms.index("llava")]
    rows = [json.loads(l) for f in sorted(glob.glob(str(LABELS / "shard_*.jsonl"))) for l in open(f)]
    books = sorted({r["image_id"].split("/")[0] for r in rows})
    val_books = set(np.random.default_rng(SPLIT_SEED).choice(books, size=max(1, int(len(books) * VAL_SHARE)), replace=False))

    boxes = []                                    # (image_id, class, yolo line, llava confirms)
    for r in rows:
        iid = r["image_id"]
        for c, x1, y1, x2, y2, sc in r["boxes"]:
            w, h = x2 - x1, y2 - y1
            if sc < BOX_THRESHOLD or w <= 0 or h <= 0:
                continue
            line = f"{CLASSES.index(c)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"
            boxes.append((iid, c, line, bool(L[pos[iid], mc.index(c)] >= LLAVA_YES)))
    keep = {"all": np.ones(len(boxes), bool), "consensus": np.array([b[3] for b in boxes])}
    rnd = np.ones(len(boxes), bool); rng = np.random.default_rng(DROP_SEED)
    for c in CLASSES:
        idx = np.array([k for k, b in enumerate(boxes) if b[1] == c])
        n_drop = int(sum(1 for k in idx if not boxes[k][3]))
        if n_drop:
            rnd[rng.choice(idx, size=n_drop, replace=False)] = False
    keep["random"] = rnd

    stats = {"pictures": len(rows), "books": len(books), "val_books": len(val_books), "box_threshold": BOX_THRESHOLD,
             "llava_yes": LLAVA_YES, "split_seed": SPLIT_SEED, "random_drop_seed": DROP_SEED, "variants": {}}
    for v in VARIANTS:
        kept = Counter(boxes[k][1] for k in np.where(keep[v])[0])
        with_boxes = len({boxes[k][0] for k in np.where(keep[v])[0]})
        stats["variants"][v] = {"boxes": int(keep[v].sum()), "boxes_per_class": {c: kept[c] for c in CLASSES},
                                "pictures_with_boxes": with_boxes, "background_pictures": len(rows) - with_boxes}
    stats["random_vs_consensus_overlap_of_dropped_boxes"] = int((~keep["random"] & ~keep["consensus"]).sum())
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(stats)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1

    for v in VARIANTS:
        lines = {r["image_id"]: [] for r in rows}
        for k in np.where(keep[v])[0]:
            lines[boxes[k][0]].append(boxes[k][2])
        root = DS / v
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True); (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for iid, out in lines.items():
            split = "val" if iid.split("/")[0] in val_books else "train"
            stem = iid.replace("/", "_"); link = root / "images" / split / f"{stem}.jpg"
            if not link.exists(): link.symlink_to(IMG / f"{iid}.jpg")
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(out) + ("\n" if out else ""))
        (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
