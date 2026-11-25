#!/usr/bin/env python3
"""Label-filter experiment, steps 2-3: train one student per training set and seed, score it on the 300.

Each student is YOLO11m from COCO weights with the rebuilt student's settings (60 epochs, patience 15,
640 px, batch 32), trained on data/label_filter/<variant> (build_label_filter_datasets.py). It is
scored on the 300 hand-annotated pictures with the same box-AP50 routine as the candidate teachers,
on all 300 and on the 164 whose pages never entered any training set.

  train    python run_label_filter_experiment.py train --variant all --seed 20260919 --device 0
  score    python run_label_filter_experiment.py score --variant all --seed 20260919 --device 0
  summary  python run_label_filter_experiment.py summary [--verify]

The summary pairs the variants by seed and reports mean AP50 per variant and per class, and the
differences consensus - all and consensus - random.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
DS = BASE / "data/label_filter"
OUT = BASE / "results/signal_rebuild/label_filter"
CS = BASE / "results/signal_rebuild/class_scores"
VARIANTS = ["all", "consensus", "random"]


def run_name(v, seed):
    return f"label_filter_{v}_s{seed}"


def train(v, seed, device):
    from ultralytics import YOLO
    YOLO(str(BASE / "yolo11m.pt")).train(
        data=str(DS / v / "data.yaml"), epochs=60, imgsz=640, batch=32, device=device,
        project=str(BASE / "runs/detect"), name=run_name(v, seed), exist_ok=True,
        patience=15, seed=seed, deterministic=True, plots=False, workers=8)


def score(v, seed, device):
    import run_teacher_candidates_on_300 as tc
    from ultralytics import YOLO
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    stems = [i.replace("/", "_") for i in truth.index]
    st = YOLO(str(BASE / "runs/detect" / run_name(v, seed) / "weights/best.pt")); names = st.names
    det = {}
    for s in stems:
        r = st.predict(str(tc.IMG / f"{s}.jpg"), conf=0.001, verbose=False, device=device)[0]
        det[s] = [(names[int(c)], [float(x) for x in b], float(sc)) for c, b, sc in
                  zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"detections_{v}_s{seed}.json").write_text(json.dumps(det))


def evaluate(det):
    import run_teacher_candidates_on_300 as tc
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    stems = [i.replace("/", "_") for i in truth.index]
    unseen = {i.replace("/", "_") for i in truth.index[truth.in_detector_train == 0]}
    res = {}
    for subset, keep in (("all_300", set(stems)), ("164_not_in_training", unseen)):
        per = {}
        for c in tc.CLASSES:
            gts = {s: [b for n, b in tc.human_boxes(s) if n == c] for s in stems if s in keep}
            per[c] = tc.ap50([(s, b, sc) for s in stems if s in keep for n, b, sc in det[s] if n == c], gts)
        vals = [x for x in per.values() if x is not None]
        res[subset] = {"per_class_ap50": {c: (round(x, 4) if x is not None else None) for c, x in per.items()},
                       "mean_ap50": round(float(np.mean(vals)), 4)}
    return res


def summary(verify):
    runs = {}
    for f in sorted(OUT.glob("detections_*_s*.json")):
        v, seed = f.stem[len("detections_"):].rsplit("_s", 1)
        runs.setdefault(v, {})[seed] = evaluate(json.loads(f.read_text()))
    out = {"experiment": "Label filter: students trained on all teacher boxes, LLaVA-confirmed boxes, or a random subset of the same size",
           "datasets": json.loads((OUT / "datasets.json").read_text())["variants"], "runs": runs, "summary": {}}
    for subset in ("all_300", "164_not_in_training"):
        s = {v: {"mean_ap50_per_seed": {k: r[subset]["mean_ap50"] for k, r in runs.get(v, {}).items()}} for v in VARIANTS}
        for v in VARIANTS:
            x = list(s[v]["mean_ap50_per_seed"].values())
            s[v]["mean"] = round(float(np.mean(x)), 4) if x else None
        for other in ("all", "random"):
            seeds = sorted(set(runs.get("consensus", {})) & set(runs.get(other, {})))
            d = [runs["consensus"][k][subset]["mean_ap50"] - runs[other][k][subset]["mean_ap50"] for k in seeds]
            s[f"consensus_minus_{other}"] = {"per_seed": [round(x, 4) for x in d],
                                            "mean": round(float(np.mean(d)), 4) if d else None}
        out["summary"][subset] = s
    path = OUT / "label_filter_results.json"
    if verify:
        same = json.loads(path.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    path.write_text(json.dumps(out, indent=2))
    for subset, s in out["summary"].items():
        print(f"\n{subset}")
        for v in VARIANTS:
            print(f"  {v:10s} mean AP50 {s[v]['mean']}  per seed {s[v]['mean_ap50_per_seed']}")
        print(f"  consensus - all    {s['consensus_minus_all']}\n  consensus - random {s['consensus_minus_random']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "score", "summary"])
    ap.add_argument("--variant", choices=VARIANTS); ap.add_argument("--seed", type=int); ap.add_argument("--device", default="0")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.step == "train": train(a.variant, a.seed, a.device)
    elif a.step == "score": score(a.variant, a.seed, a.device)
    else: return summary(a.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
