#!/usr/bin/env python3
"""Statistics over the 10-class pseudo-labels -- Prof. Mandl's experiment 4.

Which object classes the teacher labelled, how often, how large the boxes are, how they are
spread over books, and which classes it hardly ever produced. Reported per class ID with
BOTH candidate names, because the dataset yaml was renamed on 2026-02-19 after the detectors
were trained, and which name list is right is settled by the hand check in task F, not here.

Input: data/yolo11_v1_refresh/labels/{train,val}  (the set the January detectors trained on)
Output: results/signal_rebuild/pseudo_label_statistics.json
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BASE = Path("/data/brhanu/thesis_project")
DS = BASE / "data/yolo11_v1_refresh"
OUT = BASE / "results/signal_rebuild/pseudo_label_statistics.json"
NAMES_AT_TRAINING = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
NAMES_IN_YAML_NOW = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text", "hat", "furniture"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    boxes = defaultdict(list)                 # cid -> list of (area_share, aspect, split, book, picture)
    per_pic = Counter(); pics = 0; per_split = Counter()
    pics_with = defaultdict(set); books_with = defaultdict(set); all_books = set()
    cooc = Counter()
    for split in ("train", "val"):
        for lf in sorted((DS / "labels" / split).glob("*.txt")):
            rows = [l.split() for l in lf.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if len(r) == 5]
            pics += 1; per_split[split] += 1
            book = lf.stem.split("_")[0]; all_books.add(book)
            per_pic[len(rows)] += 1
            present = set()
            for r in rows:
                c = int(float(r[0])); w, h = float(r[3]), float(r[4])
                boxes[c].append((w * h, (w / h) if h else 0.0))
                pics_with[c].add(lf.stem); books_with[c].add(book); present.add(c)
            for a in present:
                for b in present:
                    if a < b: cooc[(a, b)] += 1
    total = sum(len(v) for v in boxes.values())
    classes = []
    for c in range(10):
        v = boxes.get(c, [])
        area = np.array([x[0] for x in v]) if v else np.array([0.0])
        classes.append({
            "class_id": c,
            "name_at_training_2026_01_28": NAMES_AT_TRAINING[c],
            "name_in_yaml_since_2026_02_19": NAMES_IN_YAML_NOW[c],
            "name_disputed": NAMES_AT_TRAINING[c] != NAMES_IN_YAML_NOW[c],
            "boxes": len(v),
            "share_of_all_boxes_pct": round(100 * len(v) / total, 2) if total else 0.0,
            "pictures_containing": len(pics_with[c]),
            "books_containing": len(books_with[c]),
            "box_area_share_of_picture_median": round(float(np.median(area)), 4) if v else None,
            "box_area_share_p10_p90": [round(float(np.percentile(area, 10)), 4), round(float(np.percentile(area, 90)), 4)] if v else None,
            "top_book_share_of_class_pct": (round(100 * max(Counter(p.split("_")[0] for p in pics_with[c]).values()) / len(pics_with[c]), 1) if v else None),
        })
    out = {
        "experiment": "Pseudo-label statistics (Prof. Mandl, experiment 4)",
        "source": "data/yolo11_v1_refresh labels, train + val",
        "pictures": pics, "pictures_per_split": dict(per_split), "books": len(all_books), "boxes": total,
        "pictures_with_no_box": per_pic.get(0, 0),
        "boxes_per_picture": {str(k): v for k, v in sorted(per_pic.items())},
        "classes": classes,
        "classes_never_labelled": [c["class_id"] for c in classes if c["boxes"] == 0],
        "classes_under_100_boxes": [c["class_id"] for c in classes if 0 < c["boxes"] < 100],
        "top_cooccurring_id_pairs": [{"ids": list(k), "pictures": v} for k, v in cooc.most_common(8)],
        "reading": ("Counts are by class id. Ids 2, 7, 8 and 9 carry two candidate names until task F's "
                    "hand check says which list the boxes match; the counts themselves do not change."),
    }
    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(f"pictures {pics:,} ({dict(per_split)})  books {len(all_books):,}  boxes {total:,}  no-box pictures {per_pic.get(0,0)}")
    print(f"{'id':>2} {'training name':13s} {'yaml name now':13s} {'boxes':>7} {'share':>7} {'pictures':>8} {'books':>6} {'median area':>11} {'top book %':>10}")
    for c in classes:
        print(f"{c['class_id']:>2} {c['name_at_training_2026_01_28']:13s} {c['name_in_yaml_since_2026_02_19']:13s} "
              f"{c['boxes']:>7,} {c['share_of_all_boxes_pct']:>6.2f}% {c['pictures_containing']:>8,} {c['books_containing']:>6,} "
              f"{str(c['box_area_share_of_picture_median']):>11} {str(c['top_book_share_of_class_pct']):>10}")
    print("boxes per picture:", out["boxes_per_picture"])
    print("never labelled:", out["classes_never_labelled"], "| under 100 boxes:", out["classes_under_100_boxes"])
    print("co-occurring id pairs:", out["top_cooccurring_id_pairs"][:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
