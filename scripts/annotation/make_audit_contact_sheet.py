"""
Build a numbered contact sheet for auditing AUTO-APPROVED images by eye.

The images are the same ones loaded into Label Studio project 8; this is an
alternative display path for reviewing them away from a desktop. The rater's
judgement is unchanged -- only how the image reaches them.

  python make_contact_sheet.py --batch 1 [--per-sheet 15]
"""
import argparse, csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package/task_e_auto_approved_audit"
SHEET = PKG / "labeling_sheet.csv"
OUT = PKG / "contact_sheets"
LABELS = PKG / "chat_labels.csv"

CELL, PAD, COLS = 520, 14, 3


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sheet", type=int, default=15)
    a = ap.parse_args()

    rows = list(csv.DictReader(SHEET.open()))
    done = set()
    if LABELS.exists():
        done = {int(r["cvat_id"]) for r in csv.DictReader(LABELS.open())}
    todo = [r for r in rows if int(r["cvat_id"]) not in done]

    # The sheet number follows progress rather than being passed in: passing it
    # meant "batch 1" produced different images once earlier batches were
    # recorded, silently overwriting the manifest a previous batch was keyed to.
    a.batch = len(done) // a.per_sheet + 1
    start = 0
    chunk = todo[start:start + a.per_sheet]
    if not chunk:
        print("nothing left in this batch"); return 1

    n = len(chunk)
    rowsn = (n + COLS - 1) // COLS
    W = COLS * CELL + (COLS + 1) * PAD
    HDR = 70
    H = HDR + rowsn * (CELL + 34) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 18), f"Batch {a.batch}  -  {n} images  |  {len(todo)} of 465 left"
           f"   |   all AUTO-APPROVED: reply with any WITHOUT a scene",
           fill="black", font=font(26))

    manifest = []
    for i, r in enumerate(chunk):
        col, row = i % COLS, i // COLS
        x = PAD + col * (CELL + PAD)
        y = HDR + row * (CELL + 34)
        try:
            im = Image.open(PKG / "images" / r["image_file"]).convert("RGB")
        except Exception:
            continue
        im.thumbnail((CELL, CELL))
        ox = x + (CELL - im.width) // 2
        sheet.paste(im, (ox, y))
        d.rectangle([x - 2, y - 2, x + CELL + 2, y + CELL + 2], outline="#cccccc")
        # number badge, high contrast, over a solid chip so it reads on any scan
        badge = f"{i+1}"
        d.rectangle([x + 4, y + 4, x + 4 + 26 * len(badge) + 12, y + 50], fill="#1a1a1a")
        d.text((x + 12, y + 8), badge, fill="white", font=font(34))
        d.text((x + 4, y + CELL + 6), r["ppn_id"], fill="#555555", font=font(17))
        manifest.append({"n": i + 1, "cvat_id": int(r["cvat_id"]),
                         "ppn_id": r["ppn_id"], "image_file": r["image_file"]})

    OUT.mkdir(exist_ok=True)
    png = OUT / f"batch_{a.batch:03d}.png"
    sheet.save(png, optimize=True)
    (OUT / f"batch_{a.batch:03d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {png}  ({n} images, {sheet.width}x{sheet.height})")
    print(f"remaining unlabelled: {len(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
