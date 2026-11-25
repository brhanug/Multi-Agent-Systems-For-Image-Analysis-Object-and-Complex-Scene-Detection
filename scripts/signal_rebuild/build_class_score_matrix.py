#!/usr/bin/env python3
"""The per-class score matrix: pictures x systems x classes (Phase 1, C1 output).

Joins the per-class scores of the detector (top box confidence per class), SigLIP
(sigmoid score per class prompt) and LLaVA-OneVision (P(yes) to one question per class).
Kosmos-2.5 emits text, not class scores, and is left out (decision D3).

Two layers are stored, because the raw scores are not on one scale: SigLIP's sigmoid
scores sit near zero for most pictures, while P(yes) and detector confidence use the
whole unit interval. An agreement of the form 1 - |s1 - s2| is only meaningful on a
common scale, so each score is also given as its percentile within the corpus, per
system and class. Calibrating to probabilities needs labelled data and is left to C2.

A class a system cannot score (furniture and hat for the detector) is NaN.
Each system is evaluated per class on the 300 human-annotated pictures; the detector
also on the pictures it never trained on.

    python build_class_score_matrix.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT, SUMMARY = CS / "class_score_matrix.npz", CS / "class_score_matrix_summary.json"
CLASSES = ["person", "child", "animal", "horse", "building", "weapon", "vehicle", "tree", "text", "clothing", "furniture", "hat"]
SYSTEMS = ["detector", "siglip", "llava"]


def load():
    s = np.load(CS / "siglip_class_scores.npz", allow_pickle=True)
    ids = list(s["ids"]); pos = {i: k for k, i in enumerate(ids)}
    T = np.full((len(ids), len(SYSTEMS), len(CLASSES)), np.nan, dtype=np.float64)
    sc = list(s["classes"])
    T[:, 1, :] = s["scores"][:, [sc.index(c) for c in CLASSES]]
    for line in open(CS / "yolo_class_scores.jsonl"):
        r = json.loads(line)
        for c, v in r["max_conf"].items():
            if c in CLASSES: T[pos[r["image_id"]], 0, CLASSES.index(c)] = v
    for f in sorted(glob.glob(str(CS / "llava_class_scores/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            for c, v in r["p_yes"].items():
                if v is not None: T[pos[r["image_id"]], 2, CLASSES.index(c)] = v
    return np.array(ids), T


def percentiles(T):
    P = np.full_like(T, np.nan)
    for m in range(T.shape[1]):
        for k in range(T.shape[2]):
            v = T[:, m, k]; ok = ~np.isnan(v)
            if ok.sum(): P[ok, m, k] = rankdata(v[ok]) / ok.sum()
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    ids, T = load(); P = percentiles(T)
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    pos = {i: k for k, i in enumerate(ids)}; rows = np.array([pos[i] for i in truth.index])
    unseen = truth.in_detector_train.values == 0
    # 100 of the 300 chose the LLaVA prompt style; score LLaVA on the other 200 as well
    pilot = {__import__("json").loads(l)["image_id"] for l in open(CS / "llava_class_prompt_pilot.jsonl")}
    held = np.array([i not in pilot for i in truth.index])
    per = []
    for k, c in enumerate(CLASSES):
        y = truth[c].values; row = {"class": c, "positives_in_300": int(y.sum())}
        for m, sname in enumerate(SYSTEMS):
            v = T[rows, m, k]; ok = ~np.isnan(v)
            row[f"{sname}_auroc"] = round(float(roc_auc_score(y[ok], v[ok])), 4) if ok.sum() == len(y) and 0 < y.sum() < len(y) else None
        v = T[rows, 2, k]
        if not np.isnan(v).any() and 0 < y[held].sum() < held.sum():
            row["llava_auroc_not_in_pilot"] = round(float(roc_auc_score(y[held], v[held])), 4)
        v = T[rows, 0, k]
        if not np.isnan(v).any() and 0 < y[unseen].sum() < unseen.sum():
            row["detector_auroc_unseen"] = round(float(roc_auc_score(y[unseen], v[unseen])), 4)
        per.append(row)
    coverage = {s: {c: int((~np.isnan(T[:, m, k])).sum()) for k, c in enumerate(CLASSES)} for m, s in enumerate(SYSTEMS)}
    summary = {"experiment": "Per-class score matrix (C1)", "pictures": int(len(ids)), "systems": SYSTEMS, "classes": CLASSES,
               "pictures_scored": coverage, "per_class_on_300": per,
               "scale_note": "raw scores differ in scale by system; use the percentile layer (or a calibrated layer) before comparing systems"}
    if a.verify:
        same = json.loads(SUMMARY.read_text()) == json.loads(json.dumps(summary)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    np.savez_compressed(OUT, ids=ids, systems=np.array(SYSTEMS), classes=np.array(CLASSES), raw=T.astype(np.float32), percentile=P.astype(np.float32))
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print("pictures scored per system (min over classes):", {s: min(v.values()) for s, v in coverage.items()})
    for r in per: print(" ", r)


if __name__ == "__main__":
    raise SystemExit(main())
