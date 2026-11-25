#!/usr/bin/env python3
"""Per-class detector scores over the whole corpus (Phase 1, C1).

The deployed run kept one confidence per picture and the set of class names, which
cannot feed a per-class agreement measure. This re-runs the same weights and keeps,
for every class, the highest box confidence (predictions down to 0.01, so the score
is continuous) and the number of boxes at the deployed threshold of 0.25.

Resumable: appends JSONL and skips pictures already written.

    python run_yolo_class_scores.py --device 3
"""
import argparse, glob, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
WEIGHTS = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
IMG_GLOB = str(BASE / "final_dataset/images/restored/**/*.jpg")
OUT = BASE / "results/signal_rebuild/class_scores/yolo_class_scores.jsonl"
SCORE_CONF, DEPLOYED_CONF = 0.01, 0.25


def image_id(p):
    p = Path(p)
    return f"{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="3")
    ap.add_argument("--weights", default=str(WEIGHTS), help="another detector, e.g. the rebuilt student")
    ap.add_argument("--out", default=str(OUT), help="where to write its scores")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=SCORE_CONF,
                    help="lowest box confidence kept; the per-class max is taken from what survives")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from ultralytics import YOLO
    globals()["OUT"] = Path(a.out)
    paths = sorted(glob.glob(IMG_GLOB, recursive=True))[: a.limit or None]
    done = set()
    if OUT.exists():
        for line in open(OUT):
            try: done.add(json.loads(line)["image_id"])
            except Exception: pass
    todo = [p for p in paths if image_id(p) not in done]
    print(f"corpus {len(paths)} | done {len(done)} | to run {len(todo)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights); names = model.names
    t0, n = time.time(), 0
    with open(OUT, "a") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            for p, r in zip(chunk, model.predict(source=chunk, conf=a.conf, verbose=False, device=a.device)):
                mx = {names[k]: 0.0 for k in names}; cnt = {names[k]: 0 for k in names}
                if r.boxes is not None and len(r.boxes):
                    for c, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        nm = names[int(c)]; mx[nm] = max(mx[nm], float(s)); cnt[nm] += int(s >= DEPLOYED_CONF)
                fh.write(json.dumps({"image_id": image_id(p), "max_conf": {k: round(v, 5) for k, v in mx.items()},
                                     "count_at_0.25": cnt}) + "\n")
            n += len(chunk)
            if (i // a.batch) % 25 == 0:
                fh.flush(); el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n/el:.1f} img/s", flush=True)
    print(f"DONE {n} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
