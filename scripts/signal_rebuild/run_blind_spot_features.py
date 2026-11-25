#!/usr/bin/env python3
"""Blind-spot check: can simple picture properties catch the mistakes confidence misses?

Most of the corrected pipeline's mistakes are pictures where all three agents agree and are wrong
together (faint, sparse drawings), so neither disagreement nor confidence sees them. This asks whether
properties of the picture itself -- measured from its pixels, by none of the agents -- add anything.

  pipeline  corrected + absence (0.6/0.2/0.2), scored out of fold: five book-disjoint folds over
            the 1,250 labelled pictures, each picture scored by a fit that never saw its book
  error     the pipeline's scene / no-scene call at 0.5 differs from the human label
  features  ink share (pixels darker than 0.5 of the page's range), contrast (grey-level standard
            deviation), edge density (mean gradient magnitude), colour saturation, picture area and
            aspect ratio -- all from the restored scan
  test      error ranking by confidence alone against confidence + features (logistic regression,
            nested in the same book folds), AUROC with a paired bootstrap over books; and the same
            among the pictures confidence would approve (the confident errors that pass review)

    python run_blind_spot_features.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_best_config_repeated as rb

BASE = Path("/data/brhanu/thesis_project")
IMG = BASE / "final_dataset/images/restored"
OUT = rb.SR / "blind_spot_features.json"
CACHE = rb.SR / "blind_spot_picture_features.csv"
N_FOLDS, SEED, N_BOOT = 5, 20261007, 2000
FEATS = ["ink_share", "contrast", "edge_density", "saturation", "area_mpx", "aspect"]


def picture_features(iid):
    im = Image.open(IMG / f"{iid}.jpg").convert("RGB")
    W, H = im.size
    sm = im.copy(); sm.thumbnail((512, 512))
    a = np.asarray(sm, float) / 255.0
    g = a.mean(axis=2); lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    gx, gy = np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean()
    mx, mn = a.max(axis=2), a.min(axis=2)
    return {"ink_share": float((g < lo + 0.5 * (hi - lo)).mean()), "contrast": float(g.std()),
            "edge_density": float(gx + gy), "saturation": float(((mx - mn) / np.maximum(mx, 1e-6)).mean()),
            "area_mpx": W * H / 1e6, "aspect": W / H}


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].reset_index(drop=True)
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values
    V = {**nat, "scene": scene_abs}

    if CACHE.exists():
        F = pd.read_csv(CACHE).set_index("image_id").loc[sp.ppn_id, FEATS].values
    else:
        df = pd.DataFrame([{"image_id": i, **picture_features(i)} for i in sp.ppn_id])
        if not a.verify: df.to_csv(CACHE, index=False)
        F = df[FEATS].values

    ub = np.unique(books); rng = np.random.default_rng(SEED)
    fold_of = dict(zip(rng.permutation(ub), np.arange(len(ub)) % N_FOLDS)); fold = np.array([fold_of[b] for b in books])
    saa = np.zeros(len(y)); PA = np.zeros((len(y), 3))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows[te]) for k in ("object", "scene", "vlm")])
        saa[te] = P @ rb.W; PA[te] = P
    err = ((saa >= 0.5).astype(int) != y).astype(int)
    conf_u = 1 - np.abs(saa - 0.5)                                # higher = less sure
    n_wrong = ((PA >= 0.5).astype(int) != y[:, None]).sum(axis=1)   # agents whose own call is wrong

    X0 = conf_u[:, None]; X1 = np.column_stack([conf_u, F])
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        for X, p in ((X0, p0), (X1, p1)):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], err[tr])
            p[te] = m.predict_proba(X[te])[:, 1]

    # pictures confidence would approve at a 0.05 target, threshold from out-of-fold calibration on all
    lam = rb.lam_for(conf_u, err, 0.05); approved = conf_u < lam

    def boot(mask):
        bk = np.unique(books[mask]); by = {b: np.where((books == b) & mask)[0] for b in bk}; d = []
        r = np.random.default_rng(SEED)
        for _ in range(N_BOOT):
            ii = np.concatenate([by[b] for b in r.choice(bk, len(bk))])
            if 0 < err[ii].sum() < len(ii): d.append(auc(err[ii], p1[ii]) - auc(err[ii], p0[ii]))
        d = np.array(d)
        return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                "share_above_zero": round(float((d > 0).mean()), 4)}

    full = np.ones(len(y), bool)
    coef = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X1, err)[-1].coef_[0]
    out = {"experiment": "Blind-spot check: picture properties against confidence at ranking the corrected pipeline's errors",
           "pictures": int(len(y)), "books": int(len(ub)), "errors": int(err.sum()), "features": FEATS,
           "scoring": "corrected + absence, out of fold over 5 book-disjoint folds; detectors nested in the same folds",
           "all_pictures": {"auroc_confidence": round(auc(err, p0), 4), "auroc_confidence_plus_features": round(auc(err, p1), 4),
                            "paired_gain": boot(full)},
           "among_approved_at_0.05": {"pictures": int(approved.sum()), "errors": int(err[approved].sum()),
                                      "auroc_confidence": round(auc(err[approved], p0[approved]), 4),
                                      "auroc_confidence_plus_features": round(auc(err[approved], p1[approved]), 4),
                                      "paired_gain": boot(approved)},
           "error_rate_by_ink_share_quartile": [round(float(err[(F[:, 0] >= lo) & (F[:, 0] <= hi)].mean()), 4)
                                                for lo, hi in zip(np.percentile(F[:, 0], [0, 25, 50, 75]), np.percentile(F[:, 0], [25, 50, 75, 100]))],
           "errors_by_number_of_agents_wrong": {str(k): int(((n_wrong == k) & (err == 1)).sum()) for k in range(4)},
           "pictures_with_all_three_agents_wrong": int((n_wrong == 3).sum()),
           "standardised_coefficients": {k: round(float(v), 3) for k, v in zip(["confidence_u"] + FEATS, coef)}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
