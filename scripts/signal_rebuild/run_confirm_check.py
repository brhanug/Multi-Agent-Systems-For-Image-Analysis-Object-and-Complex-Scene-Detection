#!/usr/bin/env python3
"""The confirmation check, as fixed in self_annotation_package/confirm_check/PLAN.md (commit dfb1a2a).

Does calibrating on archive pictures keep the promise on archive pictures nobody has used?

  fit         the three agents' mappings and the 0.6/0.2/0.2 combination on the whole 1,250-picture pool,
              labels under one rule (the person's labels on the 364 scene-consistency pictures;
              "unsure" keeps the old label), or as used so far
  calibrate   conformal risk control on the 399 labelled random pictures of the final check
  test        the 300 new random pictures of confirm_check/items.csv, "unsure" and unlabelled left out

Primary: line 0.5, alpha 0.05, one rule; the mean realised risk over 500 bootstrap resamples of the
calibration pictures by book (seed 20261106). Also: the calibration set as drawn with a 95% bootstrap
interval over the test pictures, the share approved, the fitted line, the labels as used so far, and
alpha 0.025.

    python run_confirm_check.py [--out PATH] [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd
from run_decision_threshold_check import fitted_line
import run_final_check as fc

CC = Path("/data/brhanu/thesis_project/self_annotation_package/confirm_check")
OUT = rb.SR / "confirm_check.json"
SEED, BOOT = 20261106, 500


def labelled(items, labels):
    m = pd.read_csv(items).merge(pd.read_csv(labels).drop_duplicates("ppn_id", keep="last")[["ppn_id", "label"]],
                                 on="ppn_id", how="left")
    counts = {"pictures": len(m), "unlabelled": int(m.label.isna().sum()), "unsure": int((m.label == "unsure").sum())}
    m = m[m.label.isin(["scene", "none"])].copy()
    m["y"] = (m.label == "scene").astype(int); m["ppn"] = m.ppn_id.str.split("/").str[0]
    counts.update(scene=int(m.y.sum()), no_scene=int((m.y == 0).sum()))
    return m.reset_index(drop=True), counts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    test, tcounts = labelled(CC / "items.csv", CC / "labels.csv")
    cal_all, person = fc.labels()
    cal = cal_all[cal_all.label != "unsure"].copy(); cal["y"] = (cal.label == "scene").astype(int)
    cal["ppn"] = cal.ppn_id.str.split("/").str[0]; cal = cal.reset_index(drop=True)
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp0 = pd.read_csv(rb.SPLIT); sp0 = sp0[sp0.ppn_id.isin(idx)].copy()
    assert not set(test.ppn_id) & (set(sp0.ppn_id) | set(cal.ppn_id)), "a test picture is in the pool or the calibration set"
    one = sp0.copy(); pl = person[person.label != "unsure"].set_index("ppn_id").label; hit = one.ppn_id.isin(pl.index)
    one.loc[hit, "has_scene"] = (one.loc[hit, "ppn_id"].map(pl) == "scene").astype(int).values

    res = {}
    for vname, sp in (("one rule", one), ("as labelled", sp0)):
        rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values
        fit = lambda r: np.column_stack([e2e.to_probability(V[k], rows, y, r) for k in ("object", "scene", "vlm")]) @ rb.W
        pp, pc, pt = fit(rows), fit(cal.ppn_id.map(idx).values), fit(test.ppn_id.map(idx).values)
        yc, yt = cal.y.values, test.y.values
        res[vname] = {}
        for ln, L in (("line 0.5", 0.5), ("fitted line", fitted_line(pp, y))):
            ec = ((pc >= L).astype(int) != yc).astype(int); uc = 1 - np.abs(pc - L)
            et = ((pt >= L).astype(int) != yt).astype(int); ut = 1 - np.abs(pt - L)
            res[vname][ln] = {"error_rate_on_test": round(float(et.mean()), 4),
                              "auroc_on_test": round(float(roc_auc_score(yt, pt)), 4)}
            for alpha in (0.05, 0.025):
                appr = sd.approve(uc, ec, ut, alpha); loss = et * appr
                rng = np.random.default_rng(SEED)
                boot = [loss[rng.integers(0, len(loss), len(loss))].mean() for _ in range(2000)]
                # the promise is an average over calibration sets: resample the calibration pictures by book
                books = cal.ppn.values; ub = np.unique(books); grp = {b: np.where(books == b)[0] for b in ub}
                rng2 = np.random.default_rng(SEED); R, A = [], []
                for _ in range(BOOT):
                    pick = np.concatenate([grp[b] for b in rng2.choice(ub, len(ub), replace=True)])
                    ap_ = sd.approve(uc[pick], ec[pick], ut, alpha); R.append(float(np.mean(et * ap_))); A.append(float(ap_.mean()))
                res[vname][ln][f"alpha {alpha}"] = {
                    "as_drawn": {"realised_risk": round(float(loss.mean()), 4),
                                 "risk_bootstrap95_over_test_pictures": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                                 "approved": round(float(appr.mean()), 4),
                                 "wrong_among_approved": round(float(et[appr].mean()), 4) if appr.any() else None},
                    "over_calibration_resamples": {"mean_realised_risk": round(float(np.mean(R)), 4),
                                                   "share_over_alpha": round(float(np.mean(np.array(R) > alpha)), 4),
                                                   "approved": rb.summarise(A)},
                    "promise_kept": bool(np.mean(R) <= alpha)}
    out = {"experiment": "Confirmation check: calibrating on archive pictures, tested on new archive pictures",
           "plan": "self_annotation_package/confirm_check/PLAN.md (commit dfb1a2a)", "seed": SEED,
           "test": tcounts, "calibration_pictures": int(len(cal)), "results": res}
    if a.verify:
        same = json.loads(a.out.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    for v, r in res.items():
        for ln, x in r.items():
            for al in ("alpha 0.05", "alpha 0.025"):
                z = x[al]
                print(f"{v:12s} {ln:11s} {al:11s} resampled risk {z['over_calibration_resamples']['mean_realised_risk']:.4f} "
                      f"(over {z['over_calibration_resamples']['share_over_alpha']:.2f}) approved {z['over_calibration_resamples']['approved']['mean']:.3f} | "
                      f"as drawn risk {z['as_drawn']['realised_risk']:.4f} {z['as_drawn']['risk_bootstrap95_over_test_pictures']} approved {z['as_drawn']['approved']:.3f} "
                      f"| kept {z['promise_kept']} | err {x['error_rate_on_test']} AUROC {x['auroc_on_test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
