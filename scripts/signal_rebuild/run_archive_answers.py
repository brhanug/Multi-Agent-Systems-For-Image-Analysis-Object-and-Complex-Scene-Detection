#!/usr/bin/env python3
"""The corrected design's three answers for every picture of the archive
(self_annotation_package/more_experiments/PLAN_archive_answers.md, commit 5e33a62). No accuracy and no
approvals are reported: those need archive labels and archive calibration.

    python run_archive_answers.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_scene_classes as rs
import run_object_classes as ro

OUT = rb.SR / "archive_answers.json"
CSV = rb.SR / "archive_answers.csv"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    ids = sorted(idx, key=idx.get); allr = np.array([idx[i] for i in ids])
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    # scene class: joint mapping on the 992
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values
    Pc = rs.to_probs(np.column_stack([V[k] for k in ("object", "scene", "vlm")]), d.ppn_id.map(idx).values, y, allr)
    # objects: Task O mappings on all 300, mean of the three agents
    R, pos = ro.scores()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    trows = np.array([pos[i] for i in truth.index]); Y = truth[ro.CLASSES].values.astype(int)
    Rx = np.concatenate([R[trows], R[[pos[i] for i in ids]]])
    Po = ro.mapped(Rx, np.vstack([Y, np.zeros((len(ids), len(ro.CLASSES)), int)]), np.arange(len(trows)), np.arange(len(Rx))).mean(1)[len(trows):]
    # scene at all: presence mappings on the 1,250-picture pool, 0.6/0.2/0.2
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)]
    Pp = np.column_stack([e2e.to_probability(V[k], sp.ppn_id.map(idx).values, sp.has_scene.values, allr)
                          for k in ("object", "scene", "vlm")]) @ rb.W
    out_pool = ~np.isin(ids, sp.ppn_id.values)
    # extra, added after the planned run: the first level fitted on the pool under one rule
    import run_final_check as fc
    _, person = fc.labels(); pl = person[person.label != "unsure"].drop_duplicates("ppn_id", keep="last").set_index("ppn_id").label
    one = sp.copy(); hit = one.ppn_id.isin(pl.index)
    one.loc[hit, "has_scene"] = (one.loc[hit, "ppn_id"].map(pl) == "scene").astype(int).values
    Pp1 = np.column_stack([e2e.to_probability(V[k], one.ppn_id.map(idx).values, one.has_scene.values, allr)
                           for k in ("object", "scene", "vlm")]) @ rb.W
    top = Pc.argmax(1); pmax = Pc.max(1); A = out_pool
    present = Po >= 0.5
    out = {"plan": "self_annotation_package/more_experiments/PLAN_archive_answers.md (commit 5e33a62)",
           "pictures": len(ids), "outside_pool": int(A.sum()),
           "scene_class": {"share_given": {c: round(float((top[A] == k).mean()), 4) for k, c in enumerate(rs.ANSWERS)},
                           "mean_top_probability": round(float(pmax[A].mean()), 4),
                           "share_top_probability_below_0.5": round(float((pmax[A] < 0.5).mean()), 4),
                           "pool_992_share_by_label": {c: round(float((y == k).mean()), 4) for k, c in enumerate(rs.ANSWERS)}},
           "objects": {"share_with_class": {c: round(float(present[A, j].mean()), 4) for j, c in enumerate(ro.CLASSES)},
                       "annotated_300_share": {c: round(float(Y[:, j].mean()), 4) for j, c in enumerate(ro.CLASSES)},
                       "classes_per_picture": round(float(present[A].sum(1).mean()), 3),
                       "annotated_300_classes_per_picture": round(float(Y.sum(1).mean()), 3)},
           "scene_at_all": {"share_scene": round(float((Pp[A] >= 0.5).mean()), 4),
                            "agreement_with_scene_class": round(float(((Pp[A] >= 0.5) == (top[A] != rs.ANSWERS.index("none"))).mean()), 4),
                            "scene_at_all_but_class_none": round(float(((Pp[A] >= 0.5) & (top[A] == rs.ANSWERS.index("none"))).mean()), 4)},
           "extra_scene_at_all_one_rule": {"share_scene": round(float((Pp1[A] >= 0.5).mean()), 4),
                                           "agreement_with_scene_class": round(float(((Pp1[A] >= 0.5) == (top[A] != rs.ANSWERS.index("none"))).mean()), 4)}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    cat = pd.DataFrame({"picture": ids, "in_pool": ~out_pool, "scene_at_all_p": np.round(Pp, 3),
                        "scene_class": [rs.ANSWERS[k] for k in top], "scene_class_p": np.round(pmax, 3),
                        **{f"p_{c}": np.round(Po[:, j], 3) for j, c in enumerate(ro.CLASSES)}})
    cat.to_csv(CSV, index=False); print(json.dumps(out, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
