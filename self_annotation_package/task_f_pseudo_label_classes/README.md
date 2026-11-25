# Task F — what do the pseudo-label class ids mean?

Twenty boxes from the 10-class pseudo-labels (`data/yolo11_v1_refresh`, 8,212 pictures) that the
January detectors were trained on.

## Why

The detector weights (trained 2026-01-28) name the classes
`0 person, 1 child, 2 horse, 3 building, 4 weapon, 5 vehicle, 6 tree, 7 clothing, 8 text, 9 animal`.
The dataset yaml was edited on 2026-02-19 to
`0 person, 1 child, 2 animal, 3 building, 4 weapon, 5 vehicle, 6 tree, 7 text, 8 hat, 9 furniture`,
but the label files were not changed. Ids **2, 7, 8, 9** therefore have two possible meanings.
Box counts per id across the whole set: `{0: 16764, 1: 77, 2: 493, 3: 68, 4: 25, 5: 6, 6: 76, 7: 31, 9: 1429}`.

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

Sampling: stratified by class id (quota `{9: 5, 2: 5, 7: 3, 0: 2, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1}`), one box per picture, seed 20260915.
