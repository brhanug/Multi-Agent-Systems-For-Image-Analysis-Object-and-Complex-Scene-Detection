#!/usr/bin/env python3
"""How completely the pre-sort found the expert "scene" pictures that break the scene rule: the author's
labels for the 324 pre-sort candidates and the 40 hidden controls, and how many expert pictures were
never relabelled.

    python run_presort_controls.py [--verify]
"""
import argparse, json

import pandas as pd

import run_best_config_repeated as rb
import run_final_check as fc
import run_scene_classes as rs

OUT = rb.SR / "presort_controls.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    cand = pd.read_csv(fc.SC / "candidates.csv"); _, person = fc.labels()
    m = cand.merge(person.drop_duplicates("ppn_id", keep="last"), on="ppn_id")
    tab = {g: m[m.group == g].label.value_counts().to_dict() for g in ("candidate", "control")}
    d, _ = rs.pool_labels(); ex = d[d.label_source == "cvat_expert_review"]
    unchecked = int((~ex.ppn_id.isin(cand.ppn_id)).sum())
    rate = tab["control"].get("none", 0) / int((m.group == "control").sum())
    out = {"labels_by_group": tab, "expert_pictures": int(len(ex)), "never_relabelled": unchecked,
           "control_rule_break_rate": round(rate, 4), "expected_unrelabelled_rule_breaks": round(rate * unchecked, 1)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
