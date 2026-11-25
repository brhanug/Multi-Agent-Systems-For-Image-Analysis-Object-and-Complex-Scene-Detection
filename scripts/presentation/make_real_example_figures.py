#!/usr/bin/env python3
"""Build the defence talk's model pictures from the pipeline's own files, so each shows a real result.

    restorer_zoom_raw.jpg / restorer_zoom_restored.jpg
        The running example's face, cropped from the original scan and from the Real-ESRGAN output
        the agents saw, at the same magnification (the restored scan is 2x, so its crop box is doubled).
    object_finder_boxes.jpg
        The object finder (the retrained student detector) run on the restored running example, with
        every box it keeps at the scoring threshold. Its boxes go to object_finder_boxes_example.json;
        --verify checks that the best box per class equals the stored class score
        (new_student_class_scores.jsonl) to 4 decimals, which ties the picture to the thesis numbers.
    text_reader_frog.jpg
        The restored frog page that Kosmos-2.5 read; its stored output is in kosmos_full/shard_*.jsonl.

    python make_real_example_figures.py            # write the pictures and the box file
    python make_real_example_figures.py --verify   # also check the boxes against the stored scores
"""
import argparse, glob, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/data/brhanu/thesis_project")
ORIG, REST = BASE / "final_dataset/images/original", BASE / "final_dataset/images/restored"
OUT = BASE / "presentation/defence_assets"
SR = BASE / "results/signal_rebuild"
EXAMPLE = "PPN1756648816/00000272_0"          # the running example (old woman, boy, hen)
FROG = "PPN1780419686/00000048_2"             # printed name in four languages
FACE = (400, 175, 600, 410)                   # crop box in original-scan pixels
DETECTOR = BASE / "runs/detect/yolo11_gdino_consensus/weights/best.pt"
CONF = 0.01                                    # same floor as run_yolo_class_scores.py
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BLUE, ORANGE = (31, 94, 168), (214, 96, 16)


def restorer():
    a = Image.open(ORIG / f"{EXAMPLE}.jpg").convert("RGB")
    b = Image.open(REST / f"{EXAMPLE}.jpg").convert("RGB")
    assert b.size == (2 * a.size[0], 2 * a.size[1]), "restored scan is not exactly 2x the original"
    x0, y0, x1, y1 = FACE
    size = ((x1 - x0) * 4, (y1 - y0) * 4)
    a.crop(FACE).resize(size, Image.BICUBIC).save(OUT / "restorer_zoom_raw.jpg", quality=90)
    b.crop((2 * x0, 2 * y0, 2 * x1, 2 * y1)).resize(size, Image.BICUBIC).save(OUT / "restorer_zoom_restored.jpg", quality=90)


def object_finder():
    from ultralytics import YOLO
    m = YOLO(str(DETECTOR))
    r = m.predict(source=str(REST / f"{EXAMPLE}.jpg"), conf=CONF, verbose=False, device="cpu")[0]
    boxes = [{"class": m.names[int(c)], "score": round(float(s), 5), "xyxy": [int(v) for v in b]}
             for b, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy())]
    boxes.sort(key=lambda d: -d["score"])
    (SR / "object_finder_boxes_example.json").write_text(json.dumps(
        {"picture": EXAMPLE, "detector": str(DETECTOR.relative_to(BASE)), "conf": CONF, "boxes": boxes}, indent=1))
    im = Image.open(REST / f"{EXAMPLE}.jpg").convert("RGB")
    d, font = ImageDraw.Draw(im), ImageFont.truetype(FONT, 96)
    for bx in boxes:
        x0, y0, x1, y1 = bx["xyxy"]; col = BLUE if bx["class"] == "person" else ORANGE
        lab = f"{bx['class']} {bx['score']:.2f}"
        d.rectangle((x0, y0, x1, y1), outline=col, width=16)
        tw, th = d.textbbox((0, 0), lab, font=font)[2:]
        tx, ty = min(x0, im.width - tw - 20), max(0, y0 - th - 22)
        d.rectangle((tx, ty, tx + tw + 20, ty + th + 16), fill=col); d.text((tx + 10, ty + 2), lab, fill="white", font=font)
    im.resize((900, round(900 * im.height / im.width)), Image.LANCZOS).save(OUT / "object_finder_boxes.jpg", quality=90)
    return boxes


def text_reader():
    im = Image.open(REST / f"{FROG}.jpg").convert("RGB")
    im.resize((700, round(700 * im.height / im.width)), Image.LANCZOS).save(OUT / "text_reader_frog.jpg", quality=90)
    for f in sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))):
        for line in open(f):
            if f'"{FROG}"' in line:
                out = json.loads(line)["kosmos_output"]
                lines = [x.strip() for x in out.split("\n") if x.strip() and set(x.strip()) - set("=")]
                return lines
    raise SystemExit(f"no stored Kosmos output for {FROG}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    restorer()
    boxes = object_finder()
    kosmos = text_reader()
    print("object finder:", [(b["class"], b["score"]) for b in boxes])
    print("text reader:", kosmos[1:4], "... 'I am a' x", kosmos.count("I am a"))
    if a.verify:
        stored = next(json.loads(l) for l in open(SR / "class_scores/new_student_class_scores.jsonl")
                      if f'"{EXAMPLE}"' in l)["max_conf"]
        best = {}
        for b in boxes: best[b["class"]] = max(best.get(b["class"], 0), b["score"])
        bad = [c for c, v in stored.items() if abs(best.get(c, 0.0) - v) > 1e-4]
        print("boxes match the stored class scores" if not bad else f"MISMATCH in {bad}: {best} vs {stored}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
