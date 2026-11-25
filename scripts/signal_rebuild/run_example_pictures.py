#!/usr/bin/env python3
"""The ten example pictures of the appendix: human labels and the corrected design's decision.

The appendix shows ten archive pictures. For each, this records the human scene labels joined on the
full picture name (book and page; the annotation file's own image_id column holds the page only and
matches other books' pages), its label in the labelled pool if it has one, and the corrected design
as fitted on the fixed split: the three agents' probabilities (mappings fitted on the 749 training
pictures), the combined probability p, the call at 0.5, and whether conformal risk control at
alpha = 0.05, calibrated on the split's 252 calibration pictures, approves it without a person.

    python run_example_pictures.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb

OUT = rb.SR / "example_pictures.json"
HSA = rb.BASE / "human_spatial_audit"
PICTURES = ["PPN1802793658/00000080_0", "PPN1837132763/00000240_0", "PPN1816431109/00000460_0",
            "PPN1859148263/00000091_1", "PPN183128149X/00000083_1", "PPN1807545539/00000142_0",
            "PPN175936214X/00000010_0", "PPN1837925887/00000046_0", "PPN1859200435/00000091_1",
            "PPN176263855X/00000017_4"]
EXPERT_ROWS = 801                      # rows of the 2,000-row file that an expert reviewed
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    import run_scene_definition_check as sd
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    tr, cal = sp[sp.split == "train"], sp[sp.split == "calibration"]
    rows_all = np.arange(max(idx.values()) + 1)
    P = np.column_stack([e2e.to_probability(V[k], tr.ppn_id.map(idx).values, tr.has_scene.values, rows_all)
                         for k in ("object", "scene", "vlm")])
    p = P @ rb.W; u = 1 - np.abs(p - 0.5)
    rc = cal.ppn_id.map(idx).values
    err_cal = ((p[rc] >= 0.5).astype(int) != cal.has_scene.values).astype(int)
    lam = rb.lam_for(u[rc], err_cal, ALPHA)

    a8 = pd.read_csv(HSA / "user_annotations_800.csv", keep_default_na=False).reset_index()
    a8["name"] = a8.raw_name.str.replace("images/", "", regex=False).str.replace(".jpg", "", regex=False)
    a3 = pd.read_csv(HSA / "user_annotations_300.csv", keep_default_na=False)
    a3["name"] = a3.filename.astype(str).str.replace(".jpg", "", regex=False)
    pool = sp.set_index("ppn_id")

    out = {"experiment": "Example pictures: human labels and the corrected design's decision",
           "alpha": ALPHA, "cut_off": round(float(lam), 4), "calibration_pictures": int(len(cal)), "pictures": []}
    for pid in PICTURES:
        name = pid.replace("/", "_", 1); r = idx[pid]
        e = a8[a8.name == name]
        expert = (e.scene_labels.iloc[0] or "none") if len(e) and int(e["index"].iloc[0]) < EXPERT_ROWS else None
        s3 = a3[a3.name == name]
        out["pictures"].append({
            "picture": pid,
            "expert_scene_label": expert,
            "second_rater_scene_label": (s3.scene.iloc[0] or "none") if len(s3) else None,
            "second_rater_boxes": int(s3.num_boxes.iloc[0]) if len(s3) else None,
            "pool_has_scene": int(pool.loc[pid, "has_scene"]) if pid in pool.index else None,
            "pool_split": pool.loc[pid, "split"] if pid in pool.index else None,
            "p_object": round(float(P[r, 0]), 3), "p_scene_matcher": round(float(P[r, 1]), 3), "p_describer": round(float(P[r, 2]), 3),
            "p": round(float(p[r]), 3), "call": "scene" if p[r] >= 0.5 else "no scene",
            "approved": bool(u[r] < lam)})
    add_classes(out, idx)
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print("cut-off", out["cut_off"])
    for q in out["pictures"]:
        print(q)
    return 0


def add_classes(out, idx):
    """The two class tasks for the ten pictures (added 8 Nov 2026).

    Scene class: the system's six-answer design (one joint mapping on all 27 agent reports, adopted after
    E6; it replaced the planned 0.6/0.2/0.2 on 9 Nov 2026), fitted
    on the fixed split's training pictures among the 992 with a known class; its single answer is
    approved under conformal risk control at alpha 0.05 calibrated on the split's calibration pictures,
    and the short list is the alpha 0.10 set. Objects: per-class mappings fitted on half of the books
    of the 300 hand-annotated pictures, the list threshold calibrated on the other half for a promise of
    at most 10% of objects missed; books that contain one of the ten pictures are left out of both."""
    import run_scene_classes as rs, run_object_classes as ro
    import run_corrected_tier1_end_to_end as e2e
    _, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({a_: i for i, a_ in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    tr = (d.split == "train").values; cal = (d.split == "calibration").values
    ex = np.array([idx[q["picture"]] for q in out["pictures"]])
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    P = rs.to_probs(Xj, rows[tr], y[tr], np.concatenate([rows, ex])); Pp, Pe = P[:len(rows)], P[len(rows):]
    pred = Pp.argmax(1); e = (pred != y).astype(int); u = 1 - Pp.max(1)
    lam = rs.lam_top1(u[cal], e[cal], 0.05)
    g = rs.set_threshold(Pp[cal], y[cal], 0.10); S = rs.make_sets(Pe, g)
    R, pos = ro.scores()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    books = np.array([i.split("/")[0] for i in truth.index]); exb = {q["picture"].split("/")[0] for q in out["pictures"]}
    keep = ~np.isin(books, list(exb)); ub = np.unique(books[keep]); rng = np.random.default_rng(20261108)
    fb = set(rng.choice(ub, len(ub) // 2, replace=False))
    trows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    fit = keep & np.isin(books, list(fb)); calo = keep & ~np.isin(books, list(fb))
    allr = np.concatenate([trows, ex]); Rr = R[allr]; Yall = np.vstack([Y, np.zeros((len(ex), len(ro.CLASSES)), int)])
    P3 = ro.mapped(Rr, Yall, np.where(fit)[0], np.arange(len(allr))).mean(1)
    t = ro.threshold(P3[:len(trows)][calo], Y[calo], 0.10)
    out["class_tasks"] = {"scene_cut_off": round(float(lam), 4), "scene_set_lambda": round(float(g), 4), "object_threshold": round(float(t), 4)}
    for k, q in enumerate(out["pictures"]):
        q["scene_class"] = rs.ANSWERS[int(Pe[k].argmax())]
        q["scene_class_probability"] = round(float(Pe[k].max()), 3)
        q["scene_class_approved"] = bool(1 - Pe[k].max() < lam)
        q["scene_short_list"] = [rs.ANSWERS[j] for j in np.argsort(-Pe[k]) if S[k, j]]
        q["object_list"] = [c for j, c in enumerate(ro.CLASSES) if P3[len(trows) + k, j] >= t]


if __name__ == "__main__":
    raise SystemExit(main())
