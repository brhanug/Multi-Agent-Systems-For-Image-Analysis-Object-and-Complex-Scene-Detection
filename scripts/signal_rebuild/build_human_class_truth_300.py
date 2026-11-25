#!/usr/bin/env python3
"""Per-picture class presence on the 300 human-annotated pictures, in one table.

The rater's boxes live in two places: the ten detector names were exported to
human_spatial_audit/labels, but furniture and hat -- on the rater's list and not the
detector's -- exist only in the Label Studio database. This joins both, so every
per-class evaluation reads the same truth.

Columns: image_id, scene (rater's category or empty), has_scene, in_detector_train,
and one 0/1 column per class in CLASSES.

    python build_human_class_truth_300.py [--verify]
"""
import argparse, json, sqlite3
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
AUD = BASE / "human_spatial_audit"
LS_DB = Path.home() / ".local/share/label-studio/label_studio.sqlite3"
OUT = BASE / "results/signal_rebuild/class_scores/human_class_truth_300.csv"
DETECTOR_NAMES = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
CLASSES = ["person", "child", "animal", "horse", "building", "weapon", "vehicle", "tree", "text", "clothing", "furniture", "hat"]
LS_ONLY = {"furniture", "hat"}


def norm(i):
    i = str(i)
    return i if "/" in i else i.replace("_", "/", 1)


def build():
    ann = pd.read_csv(AUD / "user_annotations_300.csv")
    ids = [norm(i) for i in ann.image_id]
    present = {i: set() for i in ids}
    for i in ids:
        lf = AUD / "labels" / f"{i.replace('/', '_')}.txt"
        if lf.exists():
            present[i] |= {DETECTOR_NAMES[int(float(l.split()[0]))] for l in lf.read_text().splitlines() if l.strip()}
    con = sqlite3.connect(LS_DB)
    for ds, rs in con.execute("SELECT t.data, tc.result FROM task t JOIN task_completion tc ON t.id = tc.task_id WHERE t.id <= 300"):
        res = json.loads(rs)
        for it in res if isinstance(res, list) else []:
            labs = it.get("value", {}).get("rectanglelabels") if it.get("type") == "rectanglelabels" else None
            if labs and labs[0].lower() in LS_ONLY:
                pid = norm(Path(json.loads(ds).get("image", "")).stem)
                assert pid in present, f"Label Studio picture outside the 300: {pid}"
                present[pid].add(labs[0].lower())
    train = {norm(p.stem) for p in (BASE / "data/yolo11_v2_augmented/labels/train").glob("*.txt")}
    rows = []
    for i, sc in zip(ids, ann.scene):
        r = {"image_id": i, "scene": "" if pd.isna(sc) else str(sc).strip().lower(),
             "has_scene": int(not pd.isna(sc)), "in_detector_train": int(i in train)}
        r.update({c: int(c in present[i]) for c in CLASSES})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    df = build()
    print(f"{len(df)} pictures | pictures per class:", {c: int(df[c].sum()) for c in CLASSES})
    if a.verify:
        same = pd.read_csv(OUT, keep_default_na=False).astype(str).equals(df.astype(str))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); df.to_csv(OUT, index=False); print("wrote", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
