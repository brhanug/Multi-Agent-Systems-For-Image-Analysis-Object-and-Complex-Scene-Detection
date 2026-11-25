#!/usr/bin/env python3
"""Task O of self_annotation_package/class_tasks/PLAN.md (commit d9a7fa1): which object classes a picture contains.

Pictures: the 300 hand-annotated pictures; a class is present if a human box of that class exists.
Systems: the object finder (retrained student, best-box score), the scene matcher (SigLIP class score),
the describer (LLaVA yes-strength), and their combination (mean of the three after a logistic mapping
per system and class).

200 redraws, books split 40/30/30 (fit / calibrate / test, seed 20260920). Per class: AUROC and
average precision of presence. A promise on missed objects: each picture gets the set of classes whose
probability is at least t; conformal risk control picks t so that the expected share of a picture's
present classes left out of its set is at most alpha (0.05, 0.10, 0.20).

    python run_object_classes.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/object_classes.json"
BOXES = BASE / "results/signal_rebuild/student_rebuild/teacher_candidates_300.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SYSTEMS = ["object finder", "scene matcher", "describer"]
ALPHAS, N_DRAWS, SEED = [0.05, 0.10, 0.20], 200, 20260920
GRID = np.linspace(0.0, 1.0, 1001)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def scores():
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    cols = [mc.index(c) for c in CLASSES]
    R = np.zeros((len(ids), 3, len(CLASSES)))
    R[:, 1] = np.nan_to_num(m["raw"][:, ms.index("siglip")][:, cols].astype(float))
    R[:, 2] = np.nan_to_num(m["raw"][:, ms.index("llava")][:, cols].astype(float))
    for line in open(CS / "new_student_class_scores.jsonl"):          # the retrained object finder
        r = json.loads(line); k = pos.get(r["image_id"])
        if k is None: continue
        for c, v in r["max_conf"].items():
            if c in CLASSES: R[k, 0, CLASSES.index(c)] = v
    return R, pos


def mapped(R, Y, fit, rows):
    """Logistic mapping per system and class, fitted on the fit books only."""
    P = np.zeros((len(rows), 3, len(CLASSES)))
    for s in range(3):
        for c in range(len(CLASSES)):
            y = Y[fit, c]; x = logit(R[fit, s, c]).reshape(-1, 1)
            if 0 < y.sum() < len(y):
                P[:, s, c] = LogisticRegression(C=100.0).fit(x, y).predict_proba(logit(R[rows, s, c]).reshape(-1, 1))[:, 1]
            else:
                P[:, s, c] = float(y.mean())
    return P


def fnr(S, Y):
    """Per picture: share of its present classes left out of its set (0 if it has none)."""
    npos = Y.sum(1); return np.where(npos > 0, ((Y == 1) & ~S).sum(1) / np.maximum(npos, 1), 0.0)


def threshold(P, Y, alpha):
    """Largest t whose sets {k: p_k >= t} keep the conformal bound on the miss rate (loss falls as t falls)."""
    n = len(P)
    for t in GRID[::-1]:
        if (n / (n + 1.0)) * fnr(P >= t, Y).mean() + 1.0 / (n + 1.0) <= alpha:
            return t
    return 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    R, pos = scores()
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); Y = truth[CLASSES].values.astype(int)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books)
    Rr = R[rows]
    variants = ["combined", *[f"{s} alone" for s in SYSTEMS]]
    acc = {v: {"auroc": {c: [] for c in CLASSES}, "ap": {c: [] for c in CLASSES},
               **{f"@{al}": {"miss_rate": [], "added_wrong_per_picture": [], "set_precision": [], "set_size": [],
                             "exactly_right": [], "recall_per_class": {c: [] for c in CLASSES}} for al in ALPHAS}}
           for v in variants}
    rng = np.random.default_rng(SEED)
    for _ in range(N_DRAWS):
        perm = rng.permutation(uniq); n1, n2 = int(len(uniq) * 0.4), int(len(uniq) * 0.3)
        bf, bc = set(perm[:n1]), set(perm[n1:n1 + n2])
        fit = np.array([b in bf for b in books]); cal = np.array([b in bc for b in books]); te = ~(fit | cal)
        P3 = mapped(Rr, Y, np.where(fit)[0], np.arange(len(Rr)))
        Ps = {"combined": P3.mean(1), **{f"{s} alone": P3[:, i] for i, s in enumerate(SYSTEMS)}}
        for v, P in Ps.items():
            k = acc[v]
            for ci, c in enumerate(CLASSES):
                y = Y[te, ci]
                if 0 < y.sum() < len(y):
                    k["auroc"][c].append(float(roc_auc_score(y, P[te, ci]))); k["ap"][c].append(float(average_precision_score(y, P[te, ci])))
            for al in ALPHAS:
                t = threshold(P[cal], Y[cal], al); S = P[te] >= t; Yt = Y[te]
                z = k[f"@{al}"]
                z["miss_rate"].append(float(fnr(S, Yt).mean()))
                z["added_wrong_per_picture"].append(float(((Yt == 0) & S).sum(1).mean()))
                z["set_precision"].append(float(((Yt == 1) & S).sum() / max(1, S.sum())))
                z["set_size"].append(float(S.sum(1).mean()))
                z["exactly_right"].append(float((S == (Yt == 1)).all(1).mean()))
                for ci, c in enumerate(CLASSES):
                    if Yt[:, ci].any(): z["recall_per_class"][c].append(float(S[Yt[:, ci] == 1, ci].mean()))
    r2 = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    out = {"experiment": "Task O: the eight object classes on the 300 hand-annotated pictures",
           "plan": "self_annotation_package/class_tasks/PLAN.md (commit d9a7fa1)",
           "pictures": int(len(rows)), "books": int(len(uniq)), "draws": N_DRAWS, "seed": SEED,
           "present_in_300": {c: int(Y[:, ci].sum()) for ci, c in enumerate(CLASSES)},
           "classes_per_picture": round(float(Y.sum(1).mean()), 3),
           "pictures_without_objects": int((Y.sum(1) == 0).sum()),
           "results": {}}
    for v, k in acc.items():
        o = {"auroc": {c: r2(x) for c, x in k["auroc"].items()}, "ap": {c: r2(x) for c, x in k["ap"].items()}}
        o["macro_auroc"] = round(float(np.mean([o["auroc"][c] for c in CLASSES])), 4)
        o["macro_ap"] = round(float(np.mean([o["ap"][c] for c in CLASSES])), 4)
        for al in ALPHAS:
            z = k[f"@{al}"]
            o[f"alpha {al}"] = {q: (r2(x) if q != "recall_per_class" else {c: r2(y) for c, y in x.items()}) for q, x in z.items()}
        out["results"][v] = o
    box = json.loads(BOXES.read_text())["results"]["all_300"]
    out["box_level_ap50"] = {"original object finder": box["student"]["per_class_ap50"],
                             "retrained object finder": box["new_student"]["per_class_ap50"],
                             "means": {"original": box["student"]["mean_ap50"], "retrained": box["new_student"]["mean_ap50"]}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(out["present_in_300"], out["classes_per_picture"], out["pictures_without_objects"])
    for v, o in out["results"].items():
        print(f"\n{v}: macro AUROC {o['macro_auroc']:.3f}  macro AP {o['macro_ap']:.3f}")
        print("   AUROC", o["auroc"]); print("   AP   ", o["ap"])
        for al in ALPHAS:
            z = o[f"alpha {al}"]
            print(f"   alpha {al}: miss {z['miss_rate']:.3f}  added wrong/pic {z['added_wrong_per_picture']:.2f}  precision {z['set_precision']:.3f}  "
                  f"size {z['set_size']:.2f}  exact {z['exactly_right']:.3f}  recall {z['recall_per_class']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
