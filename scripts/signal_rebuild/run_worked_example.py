#!/usr/bin/env python3
"""The running example of Chapter 6 (Sections 6.2-6.5), recomputed from the current agent scores.

The running example PPN1780419686/00000048_2 (a frog with its name printed in four languages) on the fixed split: each agent's report, the learned
weights of the three mappings (scene presence, labels as used for the main results, the scene matcher
with its "no scene" score), each agent's sum and probability, the combined probability, and the
conformal cut-off on the split's calibration pictures with its result on the test pictures. Also the
scene matcher's answer for a "no scene" score of 0.89, the value of a typical page of sheet music.

    python run_worked_example.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_best_config_repeated as rb

OUT = rb.SR / "worked_example.json"
DOG = "PPN1780419686/00000048_2"
SCENE_NAMES = ["teaching", "family", "playing", "landscape", "drawing", "no scene"]


def scene_class(e2e, idx, nat, scene_abs):
    """The running example's scene class with the system's combination (one joint mapping on all 27 reports), fitted on the
    training books of the committed split among the 992 class-labelled pictures; cut-off at alpha 0.05 on its
    calibration books. Also the fixed 0.6/0.2/0.2 combination, for comparison."""
    import run_scene_classes as rs
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    tr, cal = (d.split == "train").values, (d.split == "calibration").values
    allr = np.append(rows, idx[DOG])
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    Pj = rs.to_probs(Xj, rows[tr], y[tr], allr)
    Pk = {k: rs.to_probs(V[k], rows[tr], y[tr], allr) for k in ("object", "scene", "vlm")}
    Pf = 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"]
    res = {"train_pictures": int(tr.sum()), "calibration_pictures": int(cal.sum()),
           "describer_scene_report": {a: round(float(v), 3) for a, v in zip(rs.ANSWERS[:5], V["vlm"][idx[DOG], :5])},
           "label_one_rule": (d.set_index("ppn_id")["one rule"].get(DOG)), "split": (d.set_index("ppn_id")["split"].get(DOG))}
    for name, P in (("joint mapping", Pj), ("fixed 0.6/0.2/0.2", Pf)):
        pc = P[:-1][cal]; u = 1 - pc.max(1); e = (pc.argmax(1) != y[cal]).astype(int)
        lam = rs.lam_top1(u, e, 0.05); pdog = P[-1]
        res[name] = {"probabilities": {a: round(float(v), 3) for a, v in zip(rs.ANSWERS, pdog)},
                     "answer": rs.ANSWERS[int(pdog.argmax())], "uncertainty": round(float(1 - pdog.max()), 3),
                     "cut_off@0.05": round(float(lam), 3), "approved": bool(1 - pdog.max() < lam)}
    return res


def object_classes():
    """The running example's object list: the three mapped probabilities per class averaged, mappings fitted and the list
    threshold set at alpha 0.10 on the first of Task O's 200 redraws of the 300 hand-annotated pictures."""
    import run_object_classes as ro
    R, pos = ro.scores()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books)
    perm = np.random.default_rng(ro.SEED).permutation(uniq); n1, n2 = int(len(uniq) * 0.4), int(len(uniq) * 0.3)
    fit = np.isin(books, perm[:n1]) & (books != DOG.split("/")[0]); cal = np.isin(books, perm[n1:n1 + n2]) & (books != DOG.split("/")[0])  # never fit or calibrate on the example's own book
    Rx = np.concatenate([R[rows], R[[pos[DOG]]]])
    P = ro.mapped(Rx, np.vstack([Y, np.zeros((1, len(ro.CLASSES)), int)]), np.where(fit)[0], np.arange(len(Rx))).mean(1)
    t = ro.threshold(P[:-1][cal], Y[cal], 0.10); pd_ = P[-1]
    return {"fit_pictures": int(fit.sum()), "calibration_pictures": int(cal.sum()),
            "probabilities": {c: round(float(v), 3) for c, v in zip(ro.CLASSES, pd_)},
            "threshold@0.10": round(float(t), 3), "list": [c for c, v in zip(ro.CLASSES, pd_) if v >= t]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--picture", help="another picture than the running example (written to --out)")
    ap.add_argument("--out", help="result file for --picture"); a = ap.parse_args()
    global DOG, OUT
    if a.picture:
        DOG = a.picture; OUT = rb.SR / (a.out or ("worked_example_" + a.picture.replace("/", "_") + ".json"))
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    names = {"object": e2e.CLASSES, "scene": SCENE_NAMES, "vlm": e2e.CLASSES}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    tr, cal, te = (sp[sp.split == s] for s in ("train", "calibration", "test"))
    out = {"picture": DOG, "train_pictures": int(len(tr)), "calibration_pictures": int(len(cal)), "test_pictures": int(len(te)),
           "scene_share_in_training": round(float(tr.has_scene.mean()), 4), "agents": {}}
    d = idx[DOG]; probs = {}
    for k in ("object", "scene", "vlm"):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(V[k][tr.ppn_id.map(idx).values], tr.has_scene.values)
        z = V[k][d]; s = float(m.intercept_[0] + m.coef_[0] @ z); probs[k] = 1 / (1 + np.exp(-s))
        out["agents"][k] = {"report": {n: round(float(v), 3) for n, v in zip(names[k], z)},
                            "bias": round(float(m.intercept_[0]), 2),
                            "weights": {n: round(float(w), 2) for n, w in zip(names[k], m.coef_[0])},
                            "sum": round(s, 2), "probability": round(float(probs[k]), 2)}
        if k == "scene":
            ns = 0.89; s2 = float(m.intercept_[0] + m.coef_[0][-1] * ns)
            out["sheet_music_no_scene_score"] = ns
            out["sheet_music_scene_matcher"] = {"sum": round(s2, 2), "probability": round(float(1 / (1 + np.exp(-s2))), 2)}
    p = 0.6 * probs["object"] + 0.2 * probs["scene"] + 0.2 * probs["vlm"]
    out["combined_p"] = round(float(p), 2)
    rows = np.concatenate([cal.ppn_id.map(idx).values, te.ppn_id.map(idx).values])
    P = np.column_stack([e2e.to_probability(V[k], tr.ppn_id.map(idx).values, tr.has_scene.values, rows) for k in ("object", "scene", "vlm")]) @ rb.W
    pc, pt = P[:len(cal)], P[len(cal):]
    ec = ((pc >= 0.5).astype(int) != cal.has_scene.values).astype(int); et = ((pt >= 0.5).astype(int) != te.has_scene.values).astype(int)
    lam = rb.lam_for(1 - np.abs(pc - 0.5), ec, 0.05); ap_ = (1 - np.abs(pt - 0.5)) < lam
    n = len(cal); allowed = (0.05 - 1 / (n + 1)) * (n + 1) / n
    out["crc"] = {"alpha": 0.05, "allowance": round(1 / (n + 1), 4), "allowed_risk": round(allowed, 4),
                  "allowed_wrong": int(np.floor(allowed * n + 1e-9)), "cut_off": round(float(lam), 3),
                  "distance_from_half": round(float(1 - lam), 3), "test_approved": round(float(ap_.mean()), 4),
                  "test_risk": round(float(np.mean(et * ap_)), 4)}
    out["scene_class"] = scene_class(e2e, idx, nat, scene_abs)
    out["object_classes"] = object_classes()
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
