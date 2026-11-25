#!/usr/bin/env python3
"""Task F: settle what the pseudo-label class ids mean, and whether the boxes are any good.

The 10-class pseudo-labels in data/yolo11_v1_refresh were used to train detectors on
2026-01-28, whose weights store the class names

    0 person  1 child  2 horse  3 building  4 weapon  5 vehicle  6 tree  7 clothing  8 text  9 animal

On 2026-02-19 the dataset yaml was edited to

    0 person  1 child  2 animal  3 building  4 weapon  5 vehicle  6 tree  7 text  8 hat  9 furniture

without touching the label files. Ids 2, 7, 8 and 9 therefore have two candidate meanings,
and the 1,113 class-9 boxes are either animals or furniture. Only a person looking at the
boxes can say which. This builds a 20-box sample for that, and doubles as Prof. Mandl's
"inspect 20 pseudo-labels by hand" request.

Sampling is stratified, because a random draw is 88% person and would say nothing about the
disputed ids: 5 boxes of id 9, 5 of id 2, 3 of id 7 (id 8 has no boxes), 2 of id 0 and one
each of ids 1, 3, 4, 5, 6. One box per picture, fixed seed.

Outputs (self_annotation_package/task_f_pseudo_label_classes/):
  contact_sheets/sheet_crops.png     the 20 boxes, cropped with context, numbered
  contact_sheets/sheet_context.png   the same 20 pictures whole, every box drawn
  contact_sheets/sheet.json          what each number is
  labeling_sheet.csv                 one row per number, answer columns empty
  README.md                          the question and how to answer
"""
import csv, json, random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/data/brhanu/thesis_project")
DS = BASE / "data/yolo11_v1_refresh"
OUT = BASE / "self_annotation_package/task_f_pseudo_label_classes"
SEED = 20260915
NAMES_AT_TRAINING = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
NAMES_IN_YAML_NOW = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text", "hat", "furniture"]
QUOTA = {9: 5, 2: 5, 7: 3, 0: 2, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1}
TILE_W, TILE_H, CAP_H, COLS = 380, 300, 46, 4
HI, LO = (230, 40, 40), (40, 120, 230)


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_boxes():
    by_class = defaultdict(list)
    per_image = {}
    for split in ("train", "val"):
        for lf in sorted((DS / "labels" / split).glob("*.txt")):
            rows = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0])); x, y, w, h = map(float, parts[1:])
                rows.append((c, x, y, w, h))
            if not rows:
                continue
            key = (split, lf.stem)
            per_image[key] = rows
            for i, r in enumerate(rows):
                by_class[r[0]].append((key, i))
    return by_class, per_image


def sample(by_class):
    rng = random.Random(SEED)
    used, picks = set(), []
    for cid, k in QUOTA.items():
        pool = [p for p in by_class.get(cid, []) if p[0] not in used]
        rng.shuffle(pool)
        got = 0
        for key, idx in pool:
            if key in used:
                continue
            picks.append((cid, key, idx)); used.add(key); got += 1
            if got == k:
                break
        if got < k:
            print(f"  note: only {got} of {k} boxes available for id {cid}")
    order = list(range(len(picks)))
    rng.shuffle(order)                      # interleave so neighbours don't prime the call
    return [picks[i] for i in order]


def to_px(box, W, H):
    _, x, y, w, h = box
    return ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)


def fit(im, w, h):
    im = im.copy(); im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas, (w - im.width) // 2, (h - im.height) // 2, im.width / max(1, im.width and im.width)


def main():
    by_class, per_image = load_boxes()
    counts = {c: len(v) for c, v in sorted(by_class.items())}
    picks = sample(by_class)
    (OUT / "contact_sheets").mkdir(parents=True, exist_ok=True)
    rows_n = (len(picks) + COLS - 1) // COLS
    crops = Image.new("RGB", (COLS * TILE_W, rows_n * (TILE_H + CAP_H)), (255, 255, 255))
    ctx = Image.new("RGB", (COLS * TILE_W, rows_n * (TILE_H + CAP_H)), (255, 255, 255))
    f_big, f_small = font(17), font(13)
    manifest, problems = [], []

    for n, (cid, (split, stem), idx) in enumerate(picks, start=1):
        img_path = DS / "images" / split / f"{stem}.jpg"
        im = Image.open(img_path).convert("RGB"); W, H = im.size
        boxes = per_image[(split, stem)]
        x0, y0, x1, y1 = to_px(boxes[idx], W, H)
        if not (-1 <= x0 < x1 <= W + 1 and -1 <= y0 < y1 <= H + 1):
            problems.append(f"#{n} box outside picture: {(x0, y0, x1, y1)} in {(W, H)}")
        a, b = NAMES_AT_TRAINING[cid], NAMES_IN_YAML_NOW[cid]
        question = f"id {cid}: {a}" if a == b else f"id {cid}: {a} or {b}?"
        col, row = (n - 1) % COLS, (n - 1) // COLS
        ox, oy = col * TILE_W, row * (TILE_H + CAP_H)

        # --- crop tile: the target box with 30% context, target in red, others blue
        bw, bh = x1 - x0, y1 - y0
        pad = 0.30 * max(bw, bh, 120)
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        crop = im.crop((cx0, cy0, cx1, cy1))
        d = ImageDraw.Draw(crop)
        lw = max(2, int(max(crop.size) / 180))
        for j, bx in enumerate(boxes):
            if j == idx:
                continue
            p = to_px(bx, W, H)
            d.rectangle([p[0] - cx0, p[1] - cy0, p[2] - cx0, p[3] - cy0], outline=LO, width=max(1, lw // 2))
        d.rectangle([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], outline=HI, width=lw)
        tile, _, _, _ = fit(crop, TILE_W - 8, TILE_H - 8)
        crops.paste(tile, (ox + 4, oy + 4))
        dc = ImageDraw.Draw(crops)
        dc.text((ox + 8, oy + TILE_H + 2), f"#{n}  {question}", fill=(20, 20, 20), font=f_big)
        dc.text((ox + 8, oy + TILE_H + 24), f"red box · {stem}", fill=(110, 110, 110), font=f_small)

        # --- context tile: whole picture, every box, target thick red
        full = im.copy(); d2 = ImageDraw.Draw(full)
        lw2 = max(3, int(max(W, H) / 300))
        for j, bx in enumerate(boxes):
            p = to_px(bx, W, H)
            if j != idx:
                d2.rectangle(p, outline=LO, width=max(2, lw2 // 2))
        d2.rectangle((x0, y0, x1, y1), outline=HI, width=lw2)
        tile2, _, _, _ = fit(full, TILE_W - 8, TILE_H - 8)
        ctx.paste(tile2, (ox + 4, oy + 4))
        dx = ImageDraw.Draw(ctx)
        dx.text((ox + 8, oy + TILE_H + 2), f"#{n}  whole picture ({len(boxes)} boxes)", fill=(20, 20, 20), font=f_big)
        dx.text((ox + 8, oy + TILE_H + 24), stem, fill=(110, 110, 110), font=f_small)

        manifest.append({"n": n, "image_id": stem.replace("_", "/", 1), "split": split,
                         "label_file": f"data/yolo11_v1_refresh/labels/{split}/{stem}.txt",
                         "box_index_in_file": idx, "class_id": cid,
                         "name_at_training_2026_01_28": a, "name_in_yaml_since_2026_02_19": b,
                         "disputed": a != b, "boxes_in_picture": len(boxes),
                         "box_px": [round(v) for v in (x0, y0, x1, y1)], "picture_px": [W, H]})

    crops.save(OUT / "contact_sheets/sheet_crops.png", optimize=True)
    ctx.save(OUT / "contact_sheets/sheet_context.png", optimize=True)
    (OUT / "contact_sheets/sheet.json").write_text(json.dumps(manifest, indent=2))
    with open(OUT / "labeling_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "image_id", "class_id", "name_at_training", "name_in_yaml_now", "disputed",
                    "your_name_for_the_red_box", "box_quality_good_loose_wrong", "note"])
        for m in manifest:
            w.writerow([m["n"], m["image_id"], m["class_id"], m["name_at_training_2026_01_28"],
                        m["name_in_yaml_since_2026_02_19"], int(m["disputed"]), "", "", ""])
    (OUT / "README.md").write_text(f"""# Task F — what do the pseudo-label class ids mean?

Twenty boxes from the 10-class pseudo-labels (`data/yolo11_v1_refresh`, 8,212 pictures) that the
January detectors were trained on.

## Why

The detector weights (trained 2026-01-28) name the classes
`{", ".join(f"{i} {x}" for i, x in enumerate(NAMES_AT_TRAINING))}`.
The dataset yaml was edited on 2026-02-19 to
`{", ".join(f"{i} {x}" for i, x in enumerate(NAMES_IN_YAML_NOW))}`,
but the label files were not changed. Ids **2, 7, 8, 9** therefore have two possible meanings.
Box counts per id across the whole set: `{counts}`.

## What to look at

- `contact_sheets/sheet_crops.png` — each red box, cropped with some surrounding context.
  Other boxes in the crop are blue.
- `contact_sheets/sheet_context.png` — the same twenty pictures whole, numbered the same way.

## What to answer, for each number

1. **What is in the red box?** One of the names on the tile, or anything else (`other: …`),
   or `nothing`.
2. **Is the box good?** `good` (fits the object), `loose` (right object, poor fit) or `wrong`
   (not an object of that kind at all).

A reply like `1 furniture good, 2 horse loose, 3 person wrong, …` is enough.

Sampling: stratified by class id (quota `{QUOTA}`), one box per picture, seed {SEED}.
""")
    print(f"picked {len(manifest)} boxes from {len(per_image):,} pictures")
    print("per id picked:", {c: sum(1 for m in manifest if m['class_id'] == c) for c in sorted(QUOTA)})
    print("disputed boxes:", sum(m["disputed"] for m in manifest))
    for s in ("sheet_crops.png", "sheet_context.png"):
        p = OUT / "contact_sheets" / s; im = Image.open(p)
        print(f"{s}: {im.size[0]}x{im.size[1]} px, {p.stat().st_size/1024:.0f} KB")
    print("problems:", problems or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
