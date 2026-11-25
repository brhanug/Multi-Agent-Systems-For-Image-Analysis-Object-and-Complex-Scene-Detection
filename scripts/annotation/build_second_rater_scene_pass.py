#!/usr/bin/env python3
"""Second scene-rating pass on 150 pictures (Phase 1, A4).

Only 124 of the 300 audit pictures have two independent scene judgments, and on them
three of five categories fall below kappa 0.4 (run_per_class_human_kappa.py). This picks
150 of the other 176 -- every picture the first rater put in a rare category, then a
seeded random fill -- and lays them out for a second judgment. The first rater's label
is not shown anywhere in the package.

If the second rater is the same person as the first, wait at least two weeks and report
the result as intra-rater agreement, not inter-rater.

    python build_second_rater_scene_pass.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path("/data/brhanu/thesis_project")
C1 = BASE / "human_spatial_audit/user_annotations_800.csv"
C2 = BASE / "human_spatial_audit/user_annotations_300.csv"
IMG = BASE / "human_spatial_audit/images"
OUT = BASE / "self_annotation_package/task_h_second_rater_scenes"
N, SEED, PER_SHEET, THUMB = 150, 20260916, 15, 240
RARE = {"teaching", "family", "playing", "landscape"}


def main():
    c1 = pd.read_csv(C1); c1["cvat_id"] = c1.index
    c1["full_id"] = c1["raw_name"].str.replace("images/", "", regex=False).str.replace(".jpg", "", regex=False)
    c2 = pd.read_csv(C2)
    double = set(c2.merge(c1.loc[c1.cvat_id <= 800, ["full_id"]], left_on="image_id", right_on="full_id").image_id)
    single = c2[~c2.image_id.isin(double)].copy()
    rng = np.random.default_rng(SEED)
    cat = single.scene.fillna("none").str.lower()
    rare = list(single.image_id[cat.isin(RARE)])
    rest = list(rng.permutation(list(single.image_id[~cat.isin(RARE)])))
    picks = list(rng.permutation(rare[:N] + rest[:max(0, N - len(rare))]))
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(0, len(picks), PER_SHEET):
        part = picks[s:s + PER_SHEET]; sid = f"sheet_{s // PER_SHEET + 1:02d}"
        im = Image.new("RGB", (5 * (THUMB + 10) + 10, 3 * (THUMB + 28) + 40), "white"); d = ImageDraw.Draw(im)
        d.text((10, 10), f"{sid}: for each number, the scene category (teaching, family, playing, landscape, drawing) or none", fill="black")
        for k, iid in enumerate(part):
            t = Image.open(IMG / f"{iid}.jpg").convert("RGB"); t.thumbnail((THUMB, THUMB))
            x, y = 10 + (k % 5) * (THUMB + 10), 36 + (k // 5) * (THUMB + 28)
            im.paste(t, (x, y)); d.text((x, y + THUMB + 4), str(k + 1), fill="black")
            rows.append({"sheet": sid, "position": k + 1, "image_id": iid, "your_scene": ""})
        im.save(OUT / "sheets" / f"{sid}.png")
    pd.DataFrame(rows).to_csv(OUT / "labeling_sheet.csv", index=False)
    (OUT / "README.md").write_text(
        "# Task H - second scene rating (150 pictures)\n\n"
        "For each numbered picture, write one of: teaching, family, playing, landscape, drawing, none. "
        "Use the full picture in `human_spatial_audit/images/<image_id>.jpg` when a thumbnail is unclear.\n\n"
        "Who rates matters: a different person gives inter-rater agreement. If you rated these pictures before, "
        "wait at least two weeks and it will be reported as intra-rater agreement.\n")
    print(f"{len(picks)} pictures ({min(len(rare), N)} from rare categories), {len(range(0, len(picks), PER_SHEET))} sheets -> {OUT}")


if __name__ == "__main__":
    main()
