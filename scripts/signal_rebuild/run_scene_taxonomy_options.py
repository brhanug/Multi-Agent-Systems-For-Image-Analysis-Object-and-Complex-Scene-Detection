#!/usr/bin/env python3
"""Decision 1: what happens to human agreement on scene types if "drawing" is changed.

Prof. Mandl asked (7 Oct) for a suggestion on the scene list now that the material is better known.
"Drawing" is below chance for both systems and is where raters part most. Three ways of treating
it are compared on the existing labels, with no new labelling:

  as rated       teaching, family, playing, landscape, drawing, none
  merged         drawing counted as none (four scene types + none)
  dropped        pictures either rating calls drawing are left out (agreement on the other types)

on two samples: the 124 pictures two people rated, and the 143 pictures of the second scene rating
(the same rater twice, so consistency, not agreement). Reported: pictures, raw agreement, Cohen's
kappa with a bootstrap interval, and for the scene / no-scene decision the same measures.

    python run_scene_taxonomy_options.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import sys
BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/core_pipeline"))
import run_per_class_human_kappa as hk

OUT = BASE / "results/signal_rebuild/scene_taxonomy_options.json"
FIRST = BASE / "human_spatial_audit/user_annotations_300.csv"
TASK_H = BASE / "self_annotation_package/task_h_second_rater_scenes/labelstudio_labels.csv"
N_BOOT, SEED = 2000, 20261011


def measures(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0: return None
    rng = np.random.default_rng(SEED); ks = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1: ks.append(cohen_kappa_score(a[i], b[i]))
    return {"pictures": int(len(a)), "raw_agreement": round(float((a == b).mean()), 4),
            "kappa": round(float(cohen_kappa_score(a, b)), 4),
            "kappa_ci95": [round(float(np.percentile(ks, 2.5)), 4), round(float(np.percentile(ks, 97.5)), 4)]}


def options(a, b):
    a, b = pd.Series(a).replace("", "none").values, pd.Series(b).replace("", "none").values
    m = lambda x: np.where(x == "drawing", "none", x)
    keep = (a != "drawing") & (b != "drawing")
    s = lambda x: np.where(x == "none", "none", "scene")
    return {"as_rated": measures(a, b), "drawing_merged_into_none": measures(m(a), m(b)),
            "drawing_pictures_dropped": measures(a[keep], b[keep]),
            "scene_or_none_as_rated": measures(s(a), s(b)),
            "scene_or_none_drawing_as_none": measures(s(m(a)), s(m(b))),
            "drawing_rated_by_either": int((~keep).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    pa, pb = hk.pairs()
    f = pd.read_csv(FIRST, keep_default_na=False); f["k"] = f.image_id.astype(str).str.replace("/", "_", n=1)
    h = pd.read_csv(TASK_H, keep_default_na=False); h["k"] = h.image_id.astype(str).str.replace("/", "_", n=1)
    m = h.merge(f[["k", "scene"]].drop_duplicates("k"), on="k", suffixes=("_second", "_first"))
    out = {"experiment": "Scene list options: 'drawing' as rated, merged into none, or dropped",
           "two_people_124": options(pa, pb),
           "same_rater_twice_task_h": options(m.scene_first.str.strip().str.lower(), m.scene_second.str.strip().str.lower())}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if k == "experiment": continue
        print(k, "| drawing rated by either:", v["drawing_rated_by_either"])
        for o, r in v.items():
            if isinstance(r, dict): print(f"   {o:32s} n={r['pictures']:3d} agree {r['raw_agreement']:.3f}  kappa {r['kappa']:.3f} {r['kappa_ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
