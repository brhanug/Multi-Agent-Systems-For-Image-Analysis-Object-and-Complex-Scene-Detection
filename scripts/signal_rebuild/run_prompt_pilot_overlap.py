#!/usr/bin/env python3
"""Did choosing the describer's question on 100 of the 300 object pictures (run_llava_class_prompt_pilot.py)
favour them? Mean AUROC over the eight classes of the raw scores (the per-class mapping is monotone, so it
does not change the ranking) on the pilot's 100 pictures and on the other 200.

    python run_prompt_pilot_overlap.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb
import run_object_classes as ro

OUT = rb.SR / "prompt_pilot_overlap.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    pilot = {json.loads(l)["image_id"] for l in open(ro.CS / "llava_class_prompt_pilot.jsonl")}
    R, pos = ro.scores(); t = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    ip = t.index.isin(list(pilot)); out = {"pilot_pictures_in_300": int(ip.sum()), "mean_auroc_raw": {}}
    for name, m in (("pilot", ip), ("not in pilot", ~ip)):
        rows = np.array([pos[i] for i in t.index[m]]); Y = t[ro.CLASSES].values[m].astype(int)
        out["mean_auroc_raw"][name] = {sys_: round(float(np.mean([roc_auc_score(Y[:, c], R[rows, s, c]) for c in range(len(ro.CLASSES))
                                                                  if 0 < Y[:, c].sum() < len(Y)])), 4)
                                       for s, sys_ in ((2, "describer"), (0, "object finder"))}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
