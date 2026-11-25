#!/usr/bin/env python3
"""The corrected pipeline on pictures it has never been fitted near.

Every corrected-pipeline figure so far comes from the 1,250-picture scene pool, whose labels come
from expert review and one rater. The 300 hand-annotated pictures were labelled separately, by a
different process, and some of their books never enter the pool. Those pictures are the nearest
thing this archive has to an external test.

Eligible: pictures of the 300 whose BOOK is in neither the train nor the calibration books of the
committed split -- the books the agents' mappings and the risk threshold were fitted on. That leaves
90 pictures from 57 books. 24 of them were used to choose LLaVA's question wording, so every figure
is also reported without them.

The pipelines are exactly as committed: mappings fitted on the 749 train pictures, the risk
threshold calibrated on the 252 calibration pictures, nothing refitted here. Intervals are a
bootstrap over books (2,000 resamples), since pictures of one book are not independent.

    python run_out_of_pool_check.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "out_of_pool_check.json"
TRUTH = SR / "class_scores/human_class_truth_300.csv"
PILOT = SR / "class_scores/llava_class_prompt_pilot.jsonl"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
W = np.array([0.6, 0.2, 0.2])
ALPHAS = [0.05, 0.10, 0.15]
N_BOOT, SEED = 2000, 20260926


def label_agreement():
    """The 300 were labelled by a different process from the pool. Where a picture carries both labels,
    how often do they agree? If they define "scene" differently, this check scores the pipelines
    against a different target from the one the corrected mappings were fitted to."""
    from sklearn.metrics import cohen_kappa_score
    t = pd.read_csv(TRUTH, keep_default_na=False)[["image_id", "has_scene"]]
    sp = pd.read_csv(SPLIT)[["ppn_id", "has_scene"]]
    m = t.merge(sp, left_on="image_id", right_on="ppn_id", suffixes=("_300", "_pool"))
    a, b = m.has_scene_300.astype(int).values, m.has_scene_pool.astype(int).values
    return {"pictures_with_both_labels": int(len(m)),
            "agreement": round(float((a == b).mean()), 4),
            "cohens_kappa": round(float(cohen_kappa_score(a, b)), 4),
            "rater_no_scene_but_pool_scene": int(((a == 0) & (b == 1)).sum()),
            "rater_scene_but_pool_no_scene": int(((a == 1) & (b == 0)).sum()),
            "rater_no_scene_total": int((a == 0).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    from run_best_config_repeated import lam_for

    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(SPLIT); sp = sp[sp.ppn_id.isin(idx)]
    tr, cal = sp[sp.split == "train"], sp[sp.split == "calibration"]
    fit_books = set(tr.ppn) | set(cal.ppn)

    t = pd.read_csv(TRUTH, keep_default_na=False)
    t["book"] = t.image_id.str.split("/").str[0]
    t = t[~t.book.isin(fit_books) & t.image_id.isin(idx)].reset_index(drop=True)
    pilot = {json.loads(l)["image_id"] for l in open(PILOT)}
    y = t.has_scene.astype(int).values

    rows_tr, rows_cal = tr.ppn_id.map(idx).values, cal.ppn_id.map(idx).values
    rows_ev = t.image_id.map(idx).values
    y_tr, y_cal = tr.has_scene.values, cal.has_scene.values

    def corrected(V):
        allr = np.concatenate([rows_cal, rows_ev])
        P = np.column_stack([e2e.to_probability(V[k], rows_tr, y_tr, allr) for k in ("object", "scene", "vlm")])
        s = P @ W
        return s[:len(rows_cal)], s[len(rows_cal):]

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id")
    def deployed(ids):
        g = ag.loc[ids]
        return (0.6 * g.existing_pipeline_agent + 0.2 * g.scene_agent + 0.2 * g.vlm_agent).values, g[A4].values.std(axis=1)

    d_cal, d_cal_u = deployed(cal.ppn_id); d_ev, d_ev_u = deployed(t.image_id)
    c_cal, c_ev = corrected({**nat, "scene": scene_abs})

    pipes = {
        "deployed (disagreement triage)": (d_cal, d_ev, d_cal_u, d_ev_u),
        "corrected + absence (confidence triage)": (c_cal, c_ev, 1 - np.abs(c_cal - 0.5), 1 - np.abs(c_ev - 0.5)),
    }

    books = t.book.values; ub = np.unique(books)
    rng = np.random.default_rng(SEED)
    boot_sets = [np.concatenate([np.where(books == b)[0] for b in rng.choice(ub, len(ub), replace=True)])
                 for _ in range(N_BOOT)]

    def block(mask):
        yy = y[mask]; res = {"pictures": int(mask.sum()), "with_scene": int(yy.sum())}
        per = {}
        for name, (s_cal, s_ev, u_cal, u_ev) in pipes.items():
            err_cal = ((s_cal >= 0.5).astype(int) != y_cal).astype(int)
            err = ((s_ev >= 0.5).astype(int) != y).astype(int)
            r = {"scene_auroc": round(float(roc_auc_score(yy, s_ev[mask])), 4),
                 "error_rate": round(float(err[mask].mean()), 4), "triage": {}}
            for al in ALPHAS:
                lam = lam_for(u_cal, err_cal, al)
                auto = u_ev[mask] < lam
                r["triage"][str(al)] = {"auto_approved": round(float(auto.mean()), 4),
                                        "realised_risk": round(float(np.mean(err[mask] * auto)), 4)}
            per[name] = (r, s_ev, err)
        res["pipelines"] = {k: v[0] for k, v in per.items()}
        # paired, book-clustered bootstrap of the two differences that carry the claim
        (dn, (_, ds, de)), (cn, (_, cs, ce)) = list(per.items())
        m_idx = np.where(mask)[0]; keep = set(m_idx)
        da, dr = [], []
        for bs in boot_sets:
            bs = np.array([i for i in bs if i in keep])
            if len(np.unique(y[bs])) < 2: continue
            da.append(roc_auc_score(y[bs], cs[bs]) - roc_auc_score(y[bs], ds[bs]))
            dr.append(de[bs].mean() - ce[bs].mean())
        f = lambda xs: {"mean": round(float(np.mean(xs)), 4), "ci95": [round(float(np.percentile(xs, 2.5)), 4),
                        round(float(np.percentile(xs, 97.5)), 4)], "share_above_zero": round(float((np.array(xs) > 0).mean()), 4)}
        res["paired"] = {"scene AUROC, corrected minus deployed": f(da),
                         "error rate, deployed minus corrected": f(dr)}
        return res

    all_mask = np.ones(len(t), bool)
    no_pilot = ~t.image_id.isin(pilot).values
    out = {"experiment": "The corrected pipeline on the hand-annotated pictures outside its fitting books",
           "eligible": "pictures of the 300 whose book is in neither the train nor the calibration books",
           "nothing_refitted": "mappings from the 749 train pictures, risk threshold from the 252 calibration pictures",
           "bootstrap": f"{N_BOOT} resamples of books",
           "label_agreement_on_shared_pictures": label_agreement(),
           "all_eligible": block(all_mask),
           "excluding_llava_prompt_pilot": block(no_pilot)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for key in ("all_eligible", "excluding_llava_prompt_pilot"):
        b = out[key]
        print(f"\n{key}: {b['pictures']} pictures, {b['with_scene']} with a scene")
        for n, r in b["pipelines"].items():
            tri = "  ".join(f"a={al}: {100*r['triage'][al]['auto_approved']:.1f}% ({r['triage'][al]['realised_risk']:.3f})"
                            for al in r["triage"])
            print(f"  {n:42s} AUROC {r['scene_auroc']:.3f}  error {r['error_rate']:.3f}   {tri}")
        for k, s in b["paired"].items():
            print(f"  {k:42s} {s['mean']:+.3f}  CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  above zero in {100*s['share_above_zero']:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
