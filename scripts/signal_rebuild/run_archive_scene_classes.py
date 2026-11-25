#!/usr/bin/env python3
"""The five scene classes on random archive pictures, exactly as fixed in
self_annotation_package/archive_classes/PLAN.md (commit a250130) and its amendment of 9 Nov 2026
(written before any result: the learned combination, one joint mapping, is primary; the planned
0.6/0.2/0.2 and the describer alone are reported beside it).

Pictures: the 404 random archive pictures the author labelled "scene" in the final and confirmation
checks, with the class given on the labelling page, plus the 289 the author labelled "no scene" there;
"no scene" given on the page is applied, "unsure" is left out. Fit on the pool's 992 pictures under one
rule (planned fit; class-balanced fit as the second variant). Calibrate on the final-check pictures,
test on the confirmation-check pictures. Also every measure without risk control on all labelled
archive pictures. The 40 hidden pool controls: Cohen's kappa between the author's and the expert's class.
A class is reported only if the test pictures hold at least 5 of it.

    python run_archive_scene_classes.py            # the planned run (needs all labels)
    python run_archive_scene_classes.py --verify   # rerun and compare with the stored result
    python run_archive_scene_classes.py --dry-run  # pipeline test on random labels; prints no figures
    python run_archive_scene_classes.py --partial  # on the pictures labelled so far (added 25 Nov 2026, when labelling
                                                   # stopped at 179 of 444): scene pictures without a class are left out,
                                                   # so "no scene" is over-represented; written to archive_scene_classes_partial.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, precision_recall_fscore_support

import run_best_config_repeated as rb
import run_scene_classes as rs
import run_final_check as fc
import run_confirm_check as cc

AC = Path("/data/brhanu/thesis_project/self_annotation_package/archive_classes")
OUT = rb.SR / "archive_scene_classes.json"
ANSWERS, K, SCENES = rs.ANSWERS, rs.K, rs.ANSWERS[:5]
MIN_TEST = 5


def page_labels():
    lab = pd.read_csv(AC / "labels.csv")                       # later clicks overwrite earlier ones, as on the page
    return lab.drop_duplicates("ppn_id", keep="last").set_index("ppn_id").label


def archive_pictures(page):
    t1, _ = fc.labels(); t1 = t1[t1.label != "unsure"]
    t2, _ = cc.labelled(cc.CC / "items.csv", cc.CC / "labels.csv")
    rows = []
    for part, t in (("calibration", t1), ("test", t2)):
        for pid, lab in zip(t.ppn_id, t.label):
            if lab == "none":
                rows.append((pid, part, "none"))
            else:
                rows.append((pid, part, page.get(pid, "MISSING")))
    d = pd.DataFrame(rows, columns=["ppn_id", "part", "cls"])
    d["ppn"] = d.ppn_id.str.split("/").str[0]
    return d


def probabilities(V, rows_fit, y_fit, rows_all):
    Pk = {k: rs.to_probs(V[k], rows_fit, y_fit, rows_all) for k in ("object", "scene", "vlm")}
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    return {"joint mapping (the system)": rs.to_probs(Xj, rows_fit, y_fit, rows_all),
            "fixed 0.6/0.2/0.2 (as first planned)": 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"],
            "describer alone": Pk["vlm"]}


def measures(P, y, cal, te):
    pred = P.argmax(1); yt, pt = y[te], pred[te]
    present = [c for c in range(K) if (yt == c).sum() >= MIN_TEST]
    pr, rc, _f, sup = precision_recall_fscore_support(yt, pt, labels=range(K), zero_division=0)
    out = {"test_pictures": int(te.sum()), "test_per_class": {ANSWERS[c]: int((yt == c).sum()) for c in range(K)},
           "macro_f1_reported_classes": round(float(f1_score(yt, pt, labels=present, average="macro", zero_division=0)), 4),
           "balanced_accuracy_reported_classes": round(float(np.mean([(pt[yt == c] == c).mean() for c in present])), 4),
           "per_class": {ANSWERS[c]: {"recall": round(float(rc[c]), 4), "precision": round(float(pr[c]), 4)} for c in present}}
    u = 1 - P.max(1); e = (pred != y).astype(int); n = cal.sum()
    lam = rs.lam_top1(u[cal], e[cal], 0.05)
    feas = (n / (n + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (n + 1.0) <= 0.05
    auto = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
    out["alpha 0.05"] = {"cut_off": round(float(lam), 4), "feasible": bool(feas), "approved": round(float(auto.mean()), 4),
                         "realised_risk": round(float(np.mean(e[te] * auto)), 4),
                         "by_answer_given": {ANSWERS[c]: {"given": int((pt == c).sum()), "approved": int((auto & (pt == c)).sum()),
                                                          "approved_and_wrong": int((auto & (pt == c) & (e[te] == 1)).sum())}
                                             for c in range(K)}}
    g = rs.set_threshold(P[cal], y[cal], 0.10); S = rs.make_sets(P[te], g); cov = S[np.arange(len(yt)), yt]
    out["alpha 0.10 sets"] = {"coverage": round(float(cov.mean()), 4), "set_size": round(float(S.sum(1).mean()), 3),
                              "coverage_per_class": {ANSWERS[c]: round(float(cov[yt == c].mean()), 4) for c in present}}
    return out


def no_risk_control(P, y, rows):
    pred = P[rows].argmax(1); yy = y[rows]; present = [c for c in range(K) if (yy == c).sum() >= MIN_TEST]
    return {"pictures": int(len(yy)),
            "macro_f1_reported_classes": round(float(f1_score(yy, pred, labels=present, average="macro", zero_division=0)), 4),
            "balanced_accuracy_reported_classes": round(float(np.mean([(pred[yy == c] == c).mean() for c in present])), 4),
            "recall_per_class": {ANSWERS[c]: round(float((pred[yy == c] == c).mean()), 4) for c in present}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    out_path = OUT.with_name("archive_scene_classes_partial.json") if a.partial else OUT
    page = page_labels()
    items = pd.read_csv(AC / "items_with_source.csv")
    if a.dry_run:                                               # random labels: tests the code, shows nothing about the archive
        rng = np.random.default_rng(0)
        page = pd.Series(rng.choice(SCENES + ["none", "unsure"], len(items)), index=items.ppn_id.values)
    missing = sorted(set(items.ppn_id) - set(page.index))
    if missing and not a.dry_run and not a.partial:
        print(f"{len(missing)} of {len(items)} pictures are not labelled yet; the planned run needs all of them."); return 2
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    pool, _ = rs.pool_labels(); pool = pool[pool.ppn_id.isin(idx)].reset_index(drop=True)
    yp = pool["one rule"].map({x: i for i, x in enumerate(ANSWERS)}).values; rp = pool.ppn_id.map(idx).values
    d = archive_pictures(page)
    assert not d.ppn_id.isin(pool.ppn_id).any(), "an archive picture is in the pool"
    n_missing = int((d.cls == "MISSING").sum()); d = d[d.cls != "MISSING"].reset_index(drop=True)
    n_unsure = int((d.cls == "unsure").sum()); d = d[d.cls != "unsure"].reset_index(drop=True)
    y = d.cls.map({x: i for i, x in enumerate(ANSWERS)}).values; ra = d.ppn_id.map(idx).values
    cal, te = (d.part == "calibration").values, (d.part == "test").values
    out = {"plan": "self_annotation_package/archive_classes/PLAN.md (commit a250130), amended 9 Nov 2026",
           "dry_run_random_labels": bool(a.dry_run), "partial": bool(a.partial), "scene_pictures_without_class_left_out": n_missing,
           "items_labelled_on_page": int(len(set(items.ppn_id) & set(page.index))), "archive_pictures": int(len(d)), "unsure_left_out": n_unsure,
           "counts": {c: int((d.cls == c).sum()) for c in ANSWERS},
           "calibration_pictures": int(cal.sum()), "test_pictures": int(te.sum()), "results": {}}
    for fit in ("planned fit", "class-balanced fit"):
        rs.BALANCED = fit == "class-balanced fit"
        Ps = probabilities(V, rp, yp, ra)
        out["results"][fit] = {v: {"calibrated_on_final_check_tested_on_confirmation": measures(P, y, cal, te),
                                   "all_archive_pictures_no_risk_control": no_risk_control(P, y, np.arange(len(y)))}
                               for v, P in Ps.items()}
    rs.BALANCED = False
    ctrl = items[items.source == "pool_control"].ppn_id
    expert = pool.set_index("ppn_id")["as labelled"]
    pairs = [(expert[p], page.get(p, "MISSING")) for p in ctrl if not (a.partial and p not in page.index)]
    both = [(e_, a_) for e_, a_ in pairs if e_ in SCENES and a_ in SCENES]
    out["controls"] = {"pictures": len(pairs), "author_none": sum(a_ == "none" for _, a_ in pairs),
                       "author_unsure": sum(a_ == "unsure" for _, a_ in pairs), "both_gave_a_class": len(both),
                       "agreement": round(float(np.mean([e_ == a_ for e_, a_ in both])), 4) if both else None,
                       "kappa_five_classes": round(float(cohen_kappa_score([e_ for e_, _ in both], [a_ for _, a_ in both], labels=SCENES)), 4) if len(both) > 1 else None}
    if a.dry_run:
        print("dry run on random labels finished:", out["archive_pictures"], "pictures,", out["calibration_pictures"], "calibration,",
              out["test_pictures"], "test,", out["controls"]["pictures"], "controls; no figures shown"); return 0
    if a.verify:
        same = json.loads(out_path.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    out_path.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
