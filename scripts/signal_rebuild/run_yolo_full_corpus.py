#!/usr/bin/env python3
"""Run the trained YOLOv11 detector over all 12,110 corpus images.

The spatial detector is the one Tier 1 model that never ran corpus-wide. The manifest's
student_v3_count is zero on all 12,110 rows, which is why the fusion stage fell back to
the VQA object count and why A_object is not the detector's box count (Section 2.4). The
distilled student_iter_3 checkpoint that produced the 231-image file is no longer on
disk, so this uses the trained YOLOv11m checkpoint that is: the 10-class archival
taxonomy model at runs/detect/yolo11_v2_augmented_refresh.

This does not reconstruct the exact historical student. It answers a different and
answerable question: what the object signal looks like when a real detector is actually
run on every image, instead of being supplied by a fallback.

Resumable -- appends JSONL and skips ids already written.

Outputs: results/signal_rebuild/yolo_full_corpus.jsonl
"""
import argparse, json, glob, os, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
WEIGHTS = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
IMG_GLOB = str(BASE / "final_dataset/images/restored/**/*.jpg")
OUT = BASE / "results/signal_rebuild/yolo_full_corpus.jsonl"
CONF = 0.25


def image_id(p):
    p = Path(p)
    return f"{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None,
                        help="Write under this directory instead of results/signal_rebuild.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.out_root:
        _r = Path(args.out_root); _r.mkdir(parents=True, exist_ok=True)
        globals()['OUT'] = _r / OUT.name
        print(f'writing to {OUT}', flush=True)
    from ultralytics import YOLO
    paths = sorted(glob.glob(IMG_GLOB, recursive=True))
    if args.limit:
        paths = paths[: args.limit]

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [p for p in paths if image_id(p) not in done]
    print(f"corpus {len(paths)} | already done {len(done)} | to run {len(todo)}", flush=True)
    if not todo:
        print("nothing to do"); return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    names = model.names
    t0 = time.time(); n = 0
    with open(OUT, "a") as fh:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            res = model.predict(source=chunk, conf=CONF, verbose=False,
                                device=args.device, stream=False)
            for p, r in zip(chunk, res):
                b = r.boxes
                confs = [float(x) for x in b.conf.cpu().numpy()] if b is not None and len(b) else []
                cls = [int(x) for x in b.cls.cpu().numpy()] if b is not None and len(b) else []
                fh.write(json.dumps({
                    "image_id": image_id(p),
                    "n_box": len(confs),
                    "conf_mean": (sum(confs) / len(confs)) if confs else 0.0,
                    "conf_max": max(confs) if confs else 0.0,
                    "conf_sum": sum(confs),
                    "n_labels": len(set(cls)),
                    "labels": sorted({names[c] for c in cls}),
                }) + "\n")
            n += len(chunk)
            if (i // args.batch) % 20 == 0:
                fh.flush()
                el = time.time() - t0
                rate = n / el if el else 0
                print(f"  {n}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo)-n)/max(rate,1e-9)/60:.1f} min",
                      flush=True)
    el = time.time() - t0
    print(f"DONE {n} images in {el/60:.1f} min ({n/el:.1f} img/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
