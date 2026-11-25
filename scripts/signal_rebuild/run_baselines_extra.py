#!/usr/bin/env python3
"""Extra analysis, added after run_baselines.py (plan PLAN_baselines.md, commit 08e2412): the system against the
old system's scene-class answer without the six-picture teaching class (which the thesis does not interpret),
paired accuracy, and the class-balanced fit against the old system. Same 992 pictures and 200 redraws.

    python run_baselines_extra.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_scene_classes as rs
from run_more_experiments import redraws

OUT = rb.SR / "baselines_extra.json"
K = rs.K; T = rs.ANSWERS.index("teaching")


def bal(yt, pt, skip=()):
    return float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any() and c not in skip]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[d.ppn_id]
    saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    sm = {}
    for line in open(rb.SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sm[r["image_id"].replace("_", "/", 1)] = r["argmax"]
    old = np.where(saa >= 0.5, np.array([rs.ANSWERS.index(sm[p]) for p in d.ppn_id]), rs.ANSWERS.index("none"))
    acc = {k: [] for k in ("system bal5", "old bal5", "balanced-fit bal5", "system acc", "old acc", "balanced-fit bal6", "old bal6")}
    for tr, cal, te in redraws(d.ppn.values):
        rs.BALANCED = False; ps = rs.to_probs(Xj, rows[tr], y[tr], rows).argmax(1)
        rs.BALANCED = True; pb = rs.to_probs(Xj, rows[tr], y[tr], rows).argmax(1); rs.BALANCED = False
        yt = y[te]
        acc["system bal5"].append(bal(yt, ps[te], (T,))); acc["old bal5"].append(bal(yt, old[te], (T,)))
        acc["balanced-fit bal5"].append(bal(yt, pb[te], (T,))); acc["balanced-fit bal6"].append(bal(yt, pb[te])); acc["old bal6"].append(bal(yt, old[te]))
        acc["system acc"].append(float((ps[te] == yt).mean())); acc["old acc"].append(float((old[te] == yt).mean()))
    A = {k: np.array(v) for k, v in acc.items()}
    out = {"extra": "added after run_baselines.py", "means": {k: rb.summarise(v) for k, v in acc.items()},
           "system_minus_old": {"balanced_accuracy_without_teaching": rb.paired(A["system bal5"] - A["old bal5"]),
                                "accuracy": rb.paired(A["system acc"] - A["old acc"])},
           "balanced_fit_minus_old": {"balanced_accuracy": rb.paired(A["balanced-fit bal6"] - A["old bal6"]),
                                      "balanced_accuracy_without_teaching": rb.paired(A["balanced-fit bal5"] - A["old bal5"])}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps({k: v["mean"] for k, v in out["means"].items()}))
    for g in ("system_minus_old", "balanced_fit_minus_old"):
        for k, v in out[g].items(): print(g, k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
