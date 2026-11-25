#!/usr/bin/env python3
"""Which answers are approved on the random archive pictures? (a breakdown added after the confirmation check)

Inside the labelled pool every automatic approval is a confident "scene" call. On random archive
pictures, where far more pictures show no scene, the same design approves about 80%, so it must also
approve "no scene" calls. This breaks the confirmation check (self_annotation_package/confirm_check/
PLAN.md, commit dfb1a2a) down by the answer given, with the calibration set as drawn: one rule,
line 0.5, alpha 0.05 and 0.025. It adds no new test; the same fit, calibration and test pictures.

    python run_archive_approval_breakdown.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_scene_definition_check as sd
import run_final_check as fc
import run_confirm_check as cc

OUT = rb.SR / "archive_approval_breakdown.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    test, tcounts = cc.labelled(cc.CC / "items.csv", cc.CC / "labels.csv")
    cal_all, person = fc.labels()
    cal = cal_all[cal_all.label != "unsure"].copy(); cal["y"] = (cal.label == "scene").astype(int)
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    pl = person[person.label != "unsure"].set_index("ppn_id").label; hit = sp.ppn_id.isin(pl.index)
    sp.loc[hit, "has_scene"] = (sp.loc[hit, "ppn_id"].map(pl) == "scene").astype(int).values
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values
    fit = lambda r: np.column_stack([e2e.to_probability(V[k], rows, y, r) for k in ("object", "scene", "vlm")]) @ rb.W
    pc, pt = fit(cal.ppn_id.map(idx).values), fit(test.ppn_id.map(idx).values)
    yc, yt = cal.y.values, test.y.values
    ec = ((pc >= 0.5).astype(int) != yc).astype(int); et = ((pt >= 0.5).astype(int) != yt).astype(int)
    res = {}
    for alpha in (0.05, 0.025):
        appr = sd.approve(1 - np.abs(pc - 0.5), ec, 1 - np.abs(pt - 0.5), alpha)
        call = (pt >= 0.5).astype(int)
        r = {"approved": round(float(appr.mean()), 4), "realised_risk": round(float((et * appr).mean()), 4)}
        for name, c in (("scene calls", 1), ("no-scene calls", 0)):
            m = call == c
            r[name] = {"given": int(m.sum()), "approved": int((m & appr).sum()),
                       "share_of_approvals": round(float((m & appr).sum() / max(1, appr.sum())), 4),
                       "wrong_among_approved": round(float(et[m & appr].mean()), 4) if (m & appr).any() else None}
        r["pictures_without_scene_approved_as_no_scene"] = round(float(((yt == 0) & appr & (call == 0)).sum() / max(1, (yt == 0).sum())), 4)
        r["pictures_with_scene_approved_as_scene"] = round(float(((yt == 1) & appr & (call == 1)).sum() / max(1, (yt == 1).sum())), 4)
        res[f"alpha {alpha}"] = r
    out = {"experiment": "Breakdown of the confirmation check's approvals by the answer given (added after the test)",
           "plan_of_the_check": "self_annotation_package/confirm_check/PLAN.md (commit dfb1a2a)",
           "test": tcounts, "setting": "one rule, line 0.5, calibration set as drawn (399 random archive pictures)",
           "results": res}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
