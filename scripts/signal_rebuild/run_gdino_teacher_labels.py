#!/usr/bin/env python3
"""Student rebuild, step 2: GroundingDINO pseudo-labels for the training pool.

GroundingDINO-base, prompted with the eight frozen class names, beat the current student on every
class on the human boxes (teacher_candidates_300.json). This labels the pictures the new student may
train on: the whole archive EXCEPT the 300 hand-annotated pictures and the calibration and test
partitions of both scene pools, so no picture used to evaluate the detector or the risk control is
ever trained on. Boxes are stored with their scores (down to 0.15); the dataset builder applies the
thresholds and the LLaVA consensus filter.

Sharded and resumable.

    CUDA_VISIBLE_DEVICES=<gpu> python run_gdino_teacher_labels.py --shard i --num-shards n
"""
import argparse, glob, json, time
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
IMG = BASE / "final_dataset/images/restored"
OUTDIR = BASE / "results/signal_rebuild/student_rebuild/gdino_labels"
GDINO = "IDEA-Research/grounding-dino-base"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
PROMPT = ". ".join(CLASSES) + "."


def excluded():
    ex = set(pd.read_csv(BASE / "results/signal_rebuild/class_scores/human_class_truth_300.csv").image_id)
    for f in ("results/multi_agent/corrected_clean_holdout_split.csv", "results/multi_agent/expanded/corrected_clean_holdout_split.csv"):
        s = pd.read_csv(BASE / f); ex |= set(s.ppn_id[s.split.isin(["calibration", "test"])])
    return ex


def pool():
    ex = excluded()
    ids = sorted(f"{Path(p).parent.name}/{Path(p).stem}" for p in glob.glob(str(IMG / "**/*.jpg"), recursive=True))
    return [i for i in ids if i not in ex], len(ex)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    ids, n_ex = pool()
    mine = [i for k, i in enumerate(ids) if k % a.num_shards == a.shard]
    OUTDIR.mkdir(parents=True, exist_ok=True); out = OUTDIR / f"shard_{a.shard}.jsonl"
    done = {json.loads(l)["image_id"] for l in open(out)} if out.exists() else set()
    todo = [i for i in mine if i not in done]
    print(f"[shard {a.shard}] pool {len(ids)} (excluded {n_ex}) | mine {len(mine)} | todo {len(todo)}", flush=True)
    proc = AutoProcessor.from_pretrained(GDINO); gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).cuda().eval()
    t0 = time.time()
    with open(out, "a") as fh:
        for n, iid in enumerate(todo, 1):
            try:
                img = Image.open(IMG / f"{iid}.jpg").convert("RGB")
            except Exception:
                continue
            W, H = img.size
            inp = proc(images=img, text=PROMPT, return_tensors="pt").to("cuda")
            with torch.no_grad(): o = gd(**inp)
            g = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(H, W)])[0]
            boxes = []
            for lab, b, sc in zip(g.get("text_labels", g.get("labels")), g["boxes"].cpu().numpy(), g["scores"].cpu().numpy()):
                hit = [c for c in CLASSES if c in str(lab)]
                if hit:
                    boxes.append([hit[0], round(float(b[0] / W), 5), round(float(b[1] / H), 5), round(float(b[2] / W), 5),
                                  round(float(b[3] / H), 5), round(float(sc), 4)])
            fh.write(json.dumps({"image_id": iid, "boxes": boxes}) + "\n")
            if n % 200 == 0:
                fh.flush(); el = time.time() - t0
                print(f"[shard {a.shard}] {n}/{len(todo)}  {n/el:.2f} img/s  eta {(len(todo)-n)/(n/el)/60:.0f} min", flush=True)
    print(f"[shard {a.shard}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
