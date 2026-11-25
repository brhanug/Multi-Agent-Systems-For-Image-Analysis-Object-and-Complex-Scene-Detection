#!/usr/bin/env python3
"""Outside-the-pool check of the corrected design against a person's labels.

The corrected design is fitted and calibrated on the 1,250-picture pool only, exactly as in
best_config_repeated.json (200 book-disjoint redraws, 60/20/20, the same seed, three agents with the
"no scene" score, weights 0.6/0.2/0.2), and then scores pictures that were never in the pool. A person
labelled those pictures under the thesis rule (people, animals, a landscape or a room = scene) on the
quick-labels page, self_annotation_package/quick_labels/scene_labels.csv; "unsure" is left out.

The pictures come from two Label Studio tasks, reported separately because they were drawn differently:
  E   a random sample of the pictures the ORIGINAL design approved without review (Task E)
  A2  the unreviewed part of the catalogue's default "no scene" block (Task A2)

Pool labels for fitting, two variants:
  as labelled   the labels used throughout the thesis
  consistent    as labelled, with the vision-language-model relabelling of scene_consistency applied
                (model labels; only used for fitting, never for scoring)

Reported per variant and task, means over redraws with 2.5/97.5 percentiles: ranking (AUROC), accuracy
at line 0.5 and at the fitted line, and under conformal risk control at 0.05 the share approved, the
share of approved pictures that are wrong, and the realised risk (approved and wrong / all). "book
disjoint" repeats this only on the pictures whose book is in neither the fitting nor the calibration
part of that redraw. The original design approved every Task E picture; its error there is the share
labelled "no scene", with a Wilson 95% interval.

    python run_external_audit.py [--out PATH] [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd
import run_scene_consistency as scc
from run_decision_threshold_check import fitted_line

QL = Path("/data/brhanu/thesis_project/self_annotation_package/quick_labels")
OUT = rb.SR / "external_audit.json"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def external_labels():
    q = pd.read_csv(QL / "scene_labels.csv").drop_duplicates("key", keep="last")
    q = q[q.label.isin(["scene", "none"])].rename(columns={"key": "ppn_id"})
    E = set(pd.read_csv(QL / "E_tasks.csv").image_id); A2 = set(pd.read_csv(QL / "A2_tasks.csv").image_id)
    q["task"] = np.where(q.ppn_id.isin(E), "E", "A2")      # 5 pictures are in both; E is how they were drawn
    q["y"] = (q.label == "scene").astype(int); q["ppn"] = q.ppn_id.str.split("/").str[0]
    return q[["ppn_id", "ppn", "task", "y"]].reset_index(drop=True)


def metrics(y, p, line, a):
    c = (p >= line).astype(int); err = (c != y).astype(int)
    out = {"accuracy": float(1 - err.mean()), "approved": float(a.mean()),
           "risk": float(np.mean(err * a)),
           "wrong_among_approved": float(err[a].mean()) if a.any() else np.nan}
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan
    return out


def run(sp, idx, V, e2e, ext):
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    xr = ext.ppn_id.map(idx).values; xy = ext.y.values; xb = ext.ppn.values
    groups = {"E": ext.task.values == "E", "A2": ext.task.values == "A2", "all": np.ones(len(ext), bool)}
    acc = {}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue                                    # the same redraws as best_config_repeated
        allrows = np.concatenate([rows, xr])
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], allrows) for k in ("object", "scene", "vlm")])
        p = P @ rb.W; pp, px = p[:len(rows)], p[len(rows):]
        disjoint = np.array([b not in b_tr and b not in b_cal for b in xb])
        for lname, L in (("line 0.5", 0.5), ("fitted line", fitted_line(pp[tr], y[tr]))):
            e_cal = ((pp[cal] >= L).astype(int) != y[cal]).astype(int)
            a = sd.approve(1 - np.abs(pp[cal] - L), e_cal, 1 - np.abs(px - L), 0.05)
            for g, m in groups.items():
                for scope, mm in (("all pictures", m), ("book disjoint", m & disjoint)):
                    if mm.sum() == 0:
                        continue
                    for k, v in metrics(xy[mm], px[mm], L, a[mm]).items():
                        acc.setdefault((lname, g, scope), {}).setdefault(k, []).append(v)
                    acc[(lname, g, scope)].setdefault("pictures", []).append(int(mm.sum()))
    res = {}
    for (lname, g, scope), d in acc.items():
        r = {k: rb.summarise(np.array(v)[~np.isnan(v)]) for k, v in d.items() if k != "pictures"}
        r["pictures_mean"] = round(float(np.mean(d["pictures"])), 1)
        res.setdefault(lname, {}).setdefault(g, {})[scope] = r
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    ext = external_labels()
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    missing = int((~ext.ppn_id.isin(idx)).sum())
    ext = ext[ext.ppn_id.isin(idx)].reset_index(drop=True)
    sp0 = pd.read_csv(rb.SPLIT); sp0 = sp0[sp0.ppn_id.isin(idx)].copy()
    assert not ext.ppn_id.isin(sp0.ppn_id).any(), "an outside picture is in the pool"
    m = scc.read_labels("claude")
    variants = {}
    for name, flip in (("as labelled", set()), ("consistent", set(m.ppn_id[m.label == "none"]))):
        sp = sp0.copy(); sp.loc[sp.ppn_id.isin(flip), "has_scene"] = 0
        variants[name] = run(sp, idx, V, e2e, ext)
    e = ext[ext.task == "E"]; k = int((e.y == 0).sum())
    out = {"experiment": "Corrected design on pictures outside the pool, against a person's labels",
           "seed": rb.SEED, "alpha": 0.05,
           "labels": {t: {"pictures": int((ext.task == t).sum()), "scene": int(((ext.task == t) & (ext.y == 1)).sum()),
                          "no_scene": int(((ext.task == t) & (ext.y == 0)).sum())} for t in ("E", "A2")},
           "pictures_without_scores": missing,
           "pictures_whose_book_is_not_in_the_pool": int((~ext.ppn.isin(sp0.ppn)).sum()),
           "original_design_on_task_E": {"approved": len(e), "wrong": k, "wrong_share": round(k / len(e), 4),
                                         "wilson95": wilson(k, len(e))},
           "variants": variants}
    if a.verify:
        same = json.loads(a.out.read_text()) == json.loads(json.dumps(out, default=float))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    a.out.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=2))
    for vn, v in variants.items():
        for ln, d in v.items():
            for g in ("E", "A2", "all"):
                for scope in ("all pictures", "book disjoint"):
                    r = d[g].get(scope)
                    if r:
                        print(f"{vn:12s} {ln:11s} {g:3s} {scope:13s} n={r['pictures_mean']:6.1f} AUROC {r['auroc']['mean']:.3f} "
                              f"acc {r['accuracy']['mean']:.3f} approved {r['approved']['mean']:.3f} "
                              f"wrong|appr {r['wrong_among_approved']['mean']:.3f} risk {r['risk']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
