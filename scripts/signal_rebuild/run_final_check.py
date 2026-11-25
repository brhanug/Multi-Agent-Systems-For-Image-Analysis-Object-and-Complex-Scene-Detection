#!/usr/bin/env python3
"""The final check, as fixed in self_annotation_package/final_check/PLAN.md (commit 51428dc).

Does the corrected design keep its promise (at most 5% of pictures approved and wrong) on pictures
drawn at random from the archive, once the pool is labelled under one rule?

  test        400 pictures drawn at random from the 10,860 archive pictures outside the pool, labelled
              by a person under the thesis rule; "unsure" left out
  pool        the 1,250 pictures, labels either
                as labelled  the labels used throughout the thesis (two rules for "scene" mixed)
                one rule     the 364 scene-consistency pictures set to the person's label
                             (60 in scene_consistency/labels.csv, 304 in final_check/labels.csv);
                             "unsure" keeps the old label
  extra       one rule, with the 355 Task E/A2 pictures of the quick-labels page added to the pool
  method      three agents (object, SigLIP scene with the "no scene" score, LLaVA), 0.6/0.2/0.2,
              conformal risk control at 0.05, 200 book-disjoint redraws of the pool, seed 20260926

Per variant and line (0.5, fitted): realised risk on the test pictures (approved and wrong / all), share
approved, share of approved pictures wrong, AUROC, accuracy; means with 2.5/97.5 percentiles over
redraws, on all test pictures and on those whose book is in neither the fitting nor the calibration part;
and a 95% bootstrap interval over the test pictures (2,000 resamples, seed 20261105) for the mean
realised risk. Also: the person's agreement with the model labels on the 304, and with their own
earlier labels on the pictures labelled twice.

    python run_final_check.py [--out PATH] [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd
from run_decision_threshold_check import fitted_line
from run_external_audit import external_labels

PKG = Path("/data/brhanu/thesis_project/self_annotation_package")
FC, SC = PKG / "final_check", PKG / "scene_consistency"
OUT = rb.SR / "final_check.json"
BOOT, BOOT_SEED = 2000, 20261105


def last(path, key="ppn_id"):
    return pd.read_csv(path).drop_duplicates(key, keep="last")


def labels():
    items = pd.read_csv(FC / "items.csv"); lab = last(FC / "labels.csv")[["ppn_id", "label"]]
    m = items.merge(lab, on="ppn_id", how="left")
    if m.label.isna().any():
        raise SystemExit(f"{int(m.label.isna().sum())} of {len(m)} final-check pictures are not labelled yet")
    test = m[(m.set == "random_archive")].copy()
    person = pd.concat([last(SC / "labels.csv")[["ppn_id", "label"]], m[m.set == "scene_consistency"][["ppn_id", "label"]]])
    cand = pd.read_csv(SC / "candidates.csv")
    assert set(person.ppn_id) == set(cand.ppn_id) and len(person) == 364
    return test, person


def run(sp, idx, V, e2e, test):
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    xr = test.ppn_id.map(idx).values; xy = test.y.values; xb = test.ppn.values
    keep = {ln: {"risk": [], "approved": [], "wrong_among_approved": [], "auroc": [], "accuracy": []} for ln in ("line 0.5", "fitted line")}
    keep_dj = {ln: {k: [] for k in keep[ln]} | {"pictures": []} for ln in keep}
    loss = {ln: [] for ln in keep}                      # per redraw, per test picture: approved and wrong
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], np.concatenate([rows, xr])) for k in ("object", "scene", "vlm")])
        p = P @ rb.W; pp, px = p[:len(rows)], p[len(rows):]
        dj = np.array([b not in b_tr and b not in b_cal for b in xb])
        for ln, L in (("line 0.5", 0.5), ("fitted line", fitted_line(pp[tr], y[tr]))):
            e_cal = ((pp[cal] >= L).astype(int) != y[cal]).astype(int)
            a = sd.approve(1 - np.abs(pp[cal] - L), e_cal, 1 - np.abs(px - L), 0.05)
            err = ((px >= L).astype(int) != xy).astype(int)
            loss[ln].append(err * a)
            for d, mm in ((keep[ln], np.ones(len(xy), bool)), (keep_dj[ln], dj)):
                if mm.sum() == 0:
                    continue
                d["risk"].append(float(np.mean(err[mm] * a[mm]))); d["approved"].append(float(a[mm].mean()))
                d["wrong_among_approved"].append(float(err[mm][a[mm]].mean()) if a[mm].any() else np.nan)
                d["auroc"].append(float(roc_auc_score(xy[mm], px[mm])) if len(set(xy[mm])) == 2 else np.nan)
                d["accuracy"].append(float(1 - err[mm].mean()))
                if "pictures" in d:
                    d["pictures"].append(int(mm.sum()))
    res = {}
    brng = np.random.default_rng(BOOT_SEED)
    for ln in keep:
        Lm = np.array(loss[ln]); per_pic = Lm.mean(0)        # each picture's approved-and-wrong rate over redraws
        boot = [per_pic[brng.integers(0, len(per_pic), len(per_pic))].mean() for _ in range(BOOT)]
        summ = lambda d: {k: rb.summarise(np.array(v)[~np.isnan(np.array(v, dtype=float))]) for k, v in d.items() if k != "pictures"}
        r = {"draws": len(Lm), "test_pictures": len(xy), "mean_realised_risk": round(float(per_pic.mean()), 4),
             "risk_bootstrap95_over_pictures": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
             "share_of_redraws_over_0.05": round(float(np.mean(Lm.mean(1) > 0.05)), 4),
             "promise_kept": bool(per_pic.mean() <= 0.05),
             "all test pictures": summ(keep[ln]),
             "book disjoint": summ(keep_dj[ln]) | {"pictures_mean": round(float(np.mean(keep_dj[ln]["pictures"])), 1)}}
        res[ln] = r
    return res


def calibrate_on_archive(sp, idx, V, e2e, test, splits=500):
    """Extra, not in the plan: fit on the whole pool, calibrate on half of the random archive pictures."""
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values
    xr = test.ppn_id.map(idx).values; xy = test.y.values; xb = test.ppn.values; ub = np.unique(xb)
    fit = lambda target: np.column_stack([e2e.to_probability(V[k], rows, y, target) for k in ("object", "scene", "vlm")]) @ rb.W
    px, pp = fit(xr), fit(rows)
    out = {}
    for ln, L in (("line 0.5", 0.5), ("fitted line", fitted_line(pp, y))):
        err = ((px >= L).astype(int) != xy).astype(int); u = 1 - np.abs(px - L)
        rng = np.random.default_rng(BOOT_SEED); R, A = [], []
        for _ in range(splits):
            cb = set(rng.permutation(ub)[:len(ub) // 2]); cal = np.array([b in cb for b in xb]); te = ~cal
            a = sd.approve(u[cal], err[cal], u[te], 0.05); R.append(float(np.mean(err[te] * a))); A.append(float(a.mean()))
        out[ln] = {"error_rate_on_archive_pictures": round(float(err.mean()), 4), "mean_realised_risk": round(float(np.mean(R)), 4),
                   "share_of_splits_over_0.05": round(float(np.mean(np.array(R) > 0.05)), 4), "approved": rb.summarise(A)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    test, person = labels()
    n_unsure = int((test.label == "unsure").sum())
    test = test[test.label != "unsure"].copy(); test["y"] = (test.label == "scene").astype(int)
    test["ppn"] = test.ppn_id.str.split("/").str[0]
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    assert test.ppn_id.isin(idx).all()
    sp0 = pd.read_csv(rb.SPLIT); sp0 = sp0[sp0.ppn_id.isin(idx)].copy()
    assert not test.ppn_id.isin(sp0.ppn_id).any(), "a test picture is in the pool"

    one = sp0.copy(); pl = person[person.label != "unsure"].set_index("ppn_id").label
    hit = one.ppn_id.isin(pl.index)
    one.loc[hit, "has_scene"] = (one.loc[hit, "ppn_id"].map(pl) == "scene").astype(int).values
    ext = external_labels()
    ext = ext[~ext.ppn_id.isin(test.ppn_id)]               # a test picture never enters the pool
    plus = pd.concat([one, pd.DataFrame({"ppn_id": ext.ppn_id, "ppn": ext.ppn, "has_scene": ext.y})], ignore_index=True)

    variants = {"as labelled": run(sp0, idx, V, e2e, test), "one rule": run(one, idx, V, e2e, test),
                "one rule, with Task E/A2 in the pool (extra)": run(plus, idx, V, e2e, test)}

    extra_cal = {name: calibrate_on_archive(sp, idx, V, e2e, test) for name, sp in (("as labelled", sp0), ("one rule", one))}

    model = pd.read_csv(SC / "claude_labels.csv")[["ppn_id", "label"]]
    fc = last(FC / "labels.csv"); items = pd.read_csv(FC / "items.csv")
    k = items[items.set == "scene_consistency"].merge(fc, on="ppn_id").merge(model, on="ppn_id", suffixes=("_person", "_model"))
    k = k[k.label_person != "unsure"]
    q = last(PKG / "quick_labels/scene_labels.csv", "key").rename(columns={"key": "ppn_id"})
    rep = fc.merge(q, on="ppn_id", suffixes=("_now", "_before"))
    out = {"experiment": "Final check: the promise on pictures drawn at random from the archive",
           "plan": "self_annotation_package/final_check/PLAN.md (commit 51428dc)", "seed": rb.SEED, "alpha": 0.05,
           "test": {"pictures": int(len(test) + n_unsure), "unsure_left_out": n_unsure,
                    "scene": int(test.y.sum()), "no_scene": int((test.y == 0).sum())},
           "pool": {"no_scene_as_labelled": int((sp0.has_scene == 0).sum()), "no_scene_one_rule": int((one.has_scene == 0).sum()),
                    "pictures_with_extra": int(len(plus))},
           "person_vs_model_labels_on_304": {"pictures": int(len(k)), "agree": int((k.label_person == k.label_model).sum())},
           "person_labelled_twice": {"pictures": int(len(rep)), "same": int((rep.label_now == rep.label_before).sum())},
           "variants": variants,
           "extra_calibrated_on_random_archive_pictures": {
               "note": "Not in PLAN.md; added after the planned result was seen. Fitted on the whole pool; calibrated on "
                       "half of the random archive pictures (split by book), tested on the other half, 500 splits. The "
                       "risk follows from conformal risk control once calibration and test pictures are exchangeable; "
                       "the share approved is what this measures.",
               **extra_cal}}
    if a.verify:
        same = json.loads(a.out.read_text()) == json.loads(json.dumps(out, default=float))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    a.out.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=2))
    for vn, v in variants.items():
        for ln, r in v.items():
            t = r["all test pictures"]
            print(f"{vn:45s} {ln:11s} risk {r['mean_realised_risk']:.4f} {r['risk_bootstrap95_over_pictures']} "
                  f"over0.05 {r['share_of_redraws_over_0.05']:.2f} approved {t['approved']['mean']:.3f} "
                  f"wrong|appr {t['wrong_among_approved']['mean']:.3f} AUROC {t['auroc']['mean']:.3f} kept {r['promise_kept']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
