#!/usr/bin/env python3
"""Student rebuild, step 1: is there a better teacher than the current one?

The current student's misses are its teacher's (B3): the teacher never labelled child, weapon,
building, vehicle or text. Before any retraining, candidate teachers are scored on the 300
hand-annotated pictures against the human boxes, all with one box-AP50 routine:

  student       the deployed detector (whose teacher is the current pseudo-label set)
  gdino         GroundingDINO-base (open vocabulary), prompted with the eight class names
  gdino+llava   GroundingDINO boxes kept only for classes LLaVA confirms in the picture
                (P(yes) >= 0.5, a fixed rule -- nothing is tuned on these pictures)

AP50 is all-point interpolated precision over recall at IoU >= 0.5, per class. The pictures are
the ones the human boxes were drawn on. Detections are cached so the evaluation can be verified
without a GPU.

    CUDA_VISIBLE_DEVICES=<gpu> python run_teacher_candidates_on_300.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/student_rebuild/teacher_candidates_300.json"
CACHE = OUT.parent / "detections_300.json"
IMG = BASE / "human_spatial_audit/images"
LAB = BASE / "human_spatial_audit/labels"
STUDENT = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
NEW_STUDENT = BASE / "runs/detect/yolo11_gdino_consensus/weights/best.pt"   # the rebuilt student, once trained
NEW_CACHE = OUT.parent / "detections_300_new_student.json"
GDINO = "IDEA-Research/grounding-dino-base"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
TRAIN_ORDER = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
TO8 = {"horse": "animal", **{c: c for c in CLASSES}}
PROMPT = ". ".join(CLASSES) + "."


def human_boxes(stem):
    out = []
    f = LAB / f"{stem}.txt"
    for l in (f.read_text().splitlines() if f.exists() else []):
        p = l.split()
        if len(p) < 5: continue
        name = TO8.get(TRAIN_ORDER[int(float(p[0]))])
        if name:
            x, y, w, h = map(float, p[1:5]); out.append((name, [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
    return out


def detect(stems):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from ultralytics import YOLO
    st = YOLO(str(STUDENT)); names = st.names
    proc = AutoProcessor.from_pretrained(GDINO); gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).cuda().eval()
    det = {}
    for s in stems:
        path = IMG / f"{s}.jpg"; img = Image.open(path).convert("RGB"); W, H = img.size
        r = st.predict(str(path), conf=0.001, verbose=False, device=0)[0]
        sb = [(TO8.get(names[int(c)]), [float(v) for v in b], float(sc)) for c, b, sc in
              zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        inp = proc(images=img, text=PROMPT, return_tensors="pt").to("cuda")
        with torch.no_grad(): o = gd(**inp)
        g = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(H, W)])[0]
        gb = []
        for lab, b, sc in zip(g.get("text_labels", g.get("labels")), g["boxes"].cpu().numpy(), g["scores"].cpu().numpy()):
            hit = [c for c in CLASSES if c in str(lab)]
            if hit: gb.append((hit[0], [float(b[0] / W), float(b[1] / H), float(b[2] / W), float(b[3] / H)], float(sc)))
        det[s] = {"student": [x for x in sb if x[0]], "gdino": gb}
    return det


def detect_new_student(stems):
    from ultralytics import YOLO
    st = YOLO(str(NEW_STUDENT)); names = st.names; det = {}
    for s in stems:
        r = st.predict(str(IMG / f"{s}.jpg"), conf=0.001, verbose=False, device=0)[0]
        det[s] = [(names[int(c)], [float(v) for v in b], float(sc)) for c, b, sc in
                  zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return det


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def ap50(preds, gts):
    """preds: [(stem, box, score)], gts: {stem: [box]} for one class."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0: return None
    used = {s: [False] * len(v) for s, v in gts.items()}
    tp = []
    for s, b, _ in sorted(preds, key=lambda x: -x[2]):
        best, j = 0.0, -1
        for k, g in enumerate(gts.get(s, [])):
            v = iou(b, g)
            if v > best: best, j = v, k
        if best >= 0.5 and not used[s][j]: used[s][j] = True; tp.append(1)
        else: tp.append(0)
    tp = np.array(tp); fp = 1 - tp
    if not len(tp): return 0.0
    rec = np.cumsum(tp) / npos; prec = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    stems = [i.replace("/", "_") for i in truth.index]
    if CACHE.exists(): det = json.loads(CACHE.read_text())
    else:
        det = detect(stems); CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(det))
    cands = ["student", "gdino", "gdino+llava"]
    if NEW_STUDENT.exists():
        if NEW_CACHE.exists(): nd = json.loads(NEW_CACHE.read_text())
        else: nd = detect_new_student(stems); NEW_CACHE.write_text(json.dumps(nd))
        for s in stems: det[s]["new_student"] = nd[s]
        cands.append("new_student")
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    L = m["raw"][:, ms.index("llava")]
    confirmed = {s: {c for c in CLASSES if L[pos[s.replace('_', '/', 1)], mc.index(c)] >= 0.5} for s in stems}
    unseen = {i.replace("/", "_") for i in truth.index[truth.in_detector_train == 0]}
    res = {}
    for subset, keep in (("all_300", set(stems)), ("164_not_in_training", unseen)):
        gts = {c: {s: [b for n, b in human_boxes(s) if n == c] for s in stems if s in keep} for c in CLASSES}
        rows = {}
        for cand in cands:
            src = cand if cand in ("student", "new_student") else "gdino"
            per = {}
            for c in CLASSES:
                preds = [(s, b, sc) for s in stems if s in keep for n, b, sc in det[s][src]
                         if n == c and (cand != "gdino+llava" or c in confirmed[s])]
                per[c] = ap50(preds, gts[c])
            vals = [v for v in per.values() if v is not None]
            rows[cand] = {"per_class_ap50": {c: (round(v, 4) if v is not None else None) for c, v in per.items()},
                          "mean_ap50": round(float(np.mean(vals)), 4)}
        res[subset] = rows
    out = {"experiment": "Candidate teachers on the 300 hand-annotated pictures (student rebuild, step 1)",
           "gdino_model": GDINO, "prompt": PROMPT, "llava_rule": "keep a class only if P(yes) >= 0.5", "results": res}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for subset, rows in res.items():
        print(f"\n{subset}: mean AP50  " + "  ".join(f"{k} {v['mean_ap50']:.3f}" for k, v in rows.items()))
        print("  class      " + "".join(k.rjust(13) for k in rows))
        for c in CLASSES:
            print(f"  {c:9s}" + "".join((f"{rows[k]['per_class_ap50'][c]:.3f}" if rows[k]['per_class_ap50'][c] is not None else "—").rjust(13) for k in rows))


if __name__ == "__main__":
    raise SystemExit(main())
