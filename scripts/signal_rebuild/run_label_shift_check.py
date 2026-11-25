#!/usr/bin/env python3
"""Does label disagreement explain why the deployed pipeline ranks better out of pool?

On 90 hand-annotated pictures from books the corrected pipeline was never fitted on, the deployed
pipeline ranked scene presence better (0.929 against 0.888). The explanation offered was the target:
those pictures were labelled by a different rater, who is stricter about what counts as a scene, and
the corrected pipeline learned the scene pool's definition. That explanation was never tested.

It can be, on the 197 pictures that carry BOTH labels. Score both pipelines on the same pictures and
evaluate each against each label set:

  if the ranking order flips with the label set  -> the definition of a scene explains the gap
  if the deployed pipeline wins under both labels -> it is a real weakness of the corrected pipeline

Leakage: most of the 197 sit in the scene pool's training split, so the committed mappings have seen
them. Every picture is therefore scored out of fold: the corrected pipeline's mappings are refitted on
the whole 1,250-picture pool in five book-disjoint folds, and each picture is scored by a fit that
never saw its book. The deployed pipeline is a fixed formula and needs no refit. Intervals are a
paired bootstrap over books (2,000 resamples).

    python run_label_shift_check.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "label_shift_check.json"
TRUTH = SR / "class_scores/human_class_truth_300.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
W = np.array([0.6, 0.2, 0.2])
N_FOLDS, N_BOOT, SEED = 5, 2000, 20260929


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e

    idx, nat, _old, scene_abs = e2e.native_vectors()
    V = {**nat, "scene": scene_abs}
    pool = pd.read_csv(SPLIT); pool = pool[pool.ppn_id.isin(idx)].reset_index(drop=True)
    rows = pool.ppn_id.map(idx).values; y_pool_all = pool.has_scene.values
    books = pool.ppn.values; ub = np.unique(books)

    # out-of-fold corrected scores for every pool picture, folds by book
    rng = np.random.default_rng(SEED)
    fold_of_book = dict(zip(rng.permutation(ub), np.arange(len(ub)) % N_FOLDS))
    fold = np.array([fold_of_book[b] for b in books])
    oof = np.zeros(len(pool))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y_pool_all[tr], rows[te]) for k in ("object", "scene", "vlm")])
        oof[te] = P @ W
    pool["corrected"] = oof

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id")
    pool["deployed"] = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).loc[pool.ppn_id].values

    t = pd.read_csv(TRUTH, keep_default_na=False)[["image_id", "has_scene"]].rename(columns={"has_scene": "rater"})
    d = pool.merge(t, left_on="ppn_id", right_on="image_id")
    d["rater"] = d.rater.astype(int)
    labels = {"scene pool's labels": d.has_scene.values, "second rater's labels": d.rater.values}
    agree = (d.has_scene.values == d.rater.values)

    b = d.ppn.values; ubd = np.unique(b)
    boot = [np.concatenate([np.where(b == x)[0] for x in rng.choice(ubd, len(ubd), replace=True)]) for _ in range(N_BOOT)]

    def compare(mask, y):
        yy, dep, cor = y[mask], d.deployed.values[mask], d.corrected.values[mask]
        out = {"pictures": int(mask.sum()), "with_scene": int(yy.sum()),
               "deployed_auroc": round(float(roc_auc_score(yy, dep)), 4),
               "corrected_auroc": round(float(roc_auc_score(yy, cor)), 4)}
        m_idx = set(np.where(mask)[0]); diffs = []
        for bs in boot:
            bs = np.array([i for i in bs if i in m_idx])
            if len(np.unique(y[bs])) < 2: continue
            diffs.append(roc_auc_score(y[bs], d.corrected.values[bs]) - roc_auc_score(y[bs], d.deployed.values[bs]))
        diffs = np.array(diffs)
        out["corrected_minus_deployed"] = {"mean": round(float(diffs.mean()), 4),
                                           "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
                                           "share_above_zero": round(float((diffs > 0).mean()), 4)}
        return out

    allm = np.ones(len(d), bool)
    out = {"experiment": "Does label disagreement explain the out-of-pool ranking gap?",
           "pictures_with_both_labels": int(len(d)), "books": int(len(ubd)),
           "labels_agree_on": int(agree.sum()),
           "scoring": "corrected pipeline (with absence evidence) scored out of fold, 5 book-disjoint folds over the 1,250-picture pool; deployed pipeline as fixed",
           "bootstrap": f"{N_BOOT} resamples of books, paired",
           "all_197": {name: compare(allm, y) for name, y in labels.items()},
           "where_the_labels_agree": compare(agree, d.has_scene.values)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(f"{out['pictures_with_both_labels']} pictures from {out['books']} books; the labels agree on {out['labels_agree_on']}\n")
    f = lambda r: (f"deployed {r['deployed_auroc']:.3f}  corrected {r['corrected_auroc']:.3f}   corrected minus deployed "
                   f"{r['corrected_minus_deployed']['mean']:+.3f} [{r['corrected_minus_deployed']['ci95'][0]:+.3f}, "
                   f"{r['corrected_minus_deployed']['ci95'][1]:+.3f}], above zero in {100*r['corrected_minus_deployed']['share_above_zero']:.0f}%")
    for name, r in out["all_197"].items():
        print(f"  against the {name:22s} ({r['with_scene']} scenes)  {f(r)}")
    r = out["where_the_labels_agree"]
    print(f"  where the two labels agree ({r['pictures']} pictures)          {f(r)}")


if __name__ == "__main__":
    raise SystemExit(main())
