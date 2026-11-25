#!/usr/bin/env python3
"""Apply the redesigned signals to all 12,110 images, and test what can be removed.

Two jobs.

(1) Corpus-scale behaviour of the redesigned pipeline: what an archive would experience if
    each signal used its model's real output instead of the deployed reduction.

(2) Evidence for a recommendation. Each component is removed in turn and the system is
    re-scored, so the question "what can go?" is answered by measurement rather than by
    preference. Components tested: the derived agreement term, the spread statistic
    itself, and each of the three models.

Every variant refits its own decision threshold on the calibration partition, by balanced
accuracy -- raw error rate selects the all-positive rule in an 84.7%-positive pool.

Outputs: results/signal_rebuild/redesigned_corpus_and_recommendation.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
CRC = BASE / "results/multi_agent/corrected_clean_holdout_crc.json"
OUT = SR / "redesigned_corpus_and_recommendation.json"
CHANNEL = "softmax"          # set from --channel; see main()
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]
GRID = np.arange(0.05, 0.95, 0.005)
LGRID = np.linspace(0.001, 1.0, 1000)


def rj(paths):
    r = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(r).drop_duplicates("image_id")


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    P = pd.DataFrame(list(s.softmax)).fillna(0.0)
    s["A_scene"] = (P.max(axis=1).values if CHANNEL == "softmax"
                    else s.max_sigmoid.astype(float).values)
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm"] = txt.apply(lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAX) / len(TAX))
    d = (y[["image_id", "conf_max"]].rename(columns={"conf_max": "A_object"})
         .merge(s[["image_id", "A_scene"]], on="image_id")
         .merge(v[["image_id", "A_vlm"]], on="image_id"))
    d["A_agree"] = 1 - (d.A_object - d.A_vlm).abs()
    return d


def bal(pred, h):
    tp = int(((pred == 1) & (h == 1)).sum()); fn = int(((pred == 0) & (h == 1)).sum())
    tn = int(((pred == 0) & (h == 0)).sum()); fp = int(((pred == 1) & (h == 0)).sum())
    return ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) / 2


def score(j, cols, w, unc, tag, out):
    """Fit threshold on calibration, evaluate the uncertainty score `unc` on test."""
    saa = sum(wi * j[c].values for c, wi in zip(cols, w)) / sum(w)
    cal = j.split.values == "calibration"; te = j.split.values == "test"
    h = j.has_scene.values
    thr = float(GRID[int(np.argmax([bal((saa[cal] >= t).astype(int), h[cal]) for t in GRID]))])
    err = ((saa >= thr).astype(int) != h).astype(int)
    U = unc(j, saa)
    y = err[te]
    rec = {"components": cols, "threshold": round(thr, 3),
           "balanced_accuracy": round(bal((saa[te] >= thr).astype(int), h[te]), 4),
           "predicts_positive_pct": round(100 * float((saa[te] >= thr).mean()), 1)}
    if len(np.unique(y)) > 1:
        rec["auroc_uncertainty"] = round(float(roc_auc_score(y, U[te])), 4)
        rec["auroc_confidence"] = round(float(roc_auc_score(y, (1 - saa)[te])), 4)
    out[tag] = rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--channel", choices=["softmax", "sigmoid"], default="softmax",
                        help="SigLIP scene channel. softmax reproduces the deployed record; "
                             "sigmoid is the corrected calibrated channel (Section 2.4).")
    args = ap.parse_args()
    global CHANNEL
    CHANNEL = args.channel
    globals()['OUT'] = (OUT if CHANNEL == 'softmax'
                        else OUT.with_name(OUT.stem + '_corrected.json'))
    d = build()
    S4 = ["A_object", "A_agree", "A_scene", "A_vlm"]

    # ---------- (1) corpus-scale behaviour ----------
    d["SAA"] = 0.6 * d.A_object + 0.2 * d.A_scene + 0.2 * d.A_vlm
    d["U_SAA"] = d[S4].std(axis=1, ddof=1)
    d["cbar"] = d[S4].mean(axis=1)
    d["U_tri"] = 0.6 * d.U_SAA + 0.4 * (1 - d.cbar)
    d["conf"] = 1 - d.SAA

    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    cal = j[j.split == "calibration"]
    thr = float(GRID[int(np.argmax([bal((cal.SAA >= t).astype(int).values, cal.has_scene.values)
                                    for t in GRID]))])
    j["err"] = ((j.SAA >= thr).astype(int) != j.has_scene).astype(int)
    calj, tej = j[j.split == "calibration"], j[j.split == "test"]

    def crc_at(U_cal, E_cal, U_te, E_te, alpha=0.05):
        n = len(U_cal)
        r = np.array([np.mean(E_cal * (U_cal < g)) for g in LGRID])
        ok = (n / (n + 1.0)) * r + (1.0 / (n + 1.0)) <= alpha
        lam = float(np.max(LGRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
        auto = U_te < lam
        caught, tot = int(E_te[~auto].sum()), int(E_te.sum())
        return {"lambda": round(lam, 4), "auto_approval_pct": round(100 * float(auto.mean()), 2),
                "error_recall_pct": round(100 * caught / tot, 2) if tot else None,
                "test_risk": round(float(np.mean(E_te * auto)), 4),
                "bound_satisfied": bool(np.mean(E_te * auto) <= alpha)}

    out = {"experiment": "Redesigned signals at corpus scale, and what can be removed",
           "n_corpus": int(len(d)), "decision_threshold": round(thr, 3),
           "corpus_scale": {}}
    for tag, U in [("triage on spread (as designed)", "U_tri"), ("triage on confidence", "conf")]:
        crc = crc_at(calj[U if U != "conf" else "conf"].values, calj.err.values,
                     tej[U if U != "conf" else "conf"].values, tej.err.values)
        lam = crc["lambda"]
        auto = (d[U] < lam)
        out["corpus_scale"][tag] = {
            **crc,
            "corpus_auto_approved": int(auto.sum()),
            "corpus_auto_approval_pct": round(100 * float(auto.mean()), 2),
            "corpus_escalated": int((~auto).sum())}

    # ---------- (2) what can be removed ----------
    rem = {}
    spread = lambda cols: (lambda jj, saa: jj[cols].std(axis=1, ddof=1).values)
    conf_only = lambda jj, saa: 1 - saa
    score(j, ["A_object", "A_scene", "A_vlm"], [0.6, 0.2, 0.2], spread(S4),
          "full system (4-signal spread)", rem)
    score(j, ["A_object", "A_scene", "A_vlm"], [0.6, 0.2, 0.2],
          spread(["A_object", "A_scene", "A_vlm"]), "remove A_agree", rem)
    score(j, ["A_object", "A_scene", "A_vlm"], [0.6, 0.2, 0.2], conf_only,
          "remove the spread entirely (confidence only)", rem)
    score(j, ["A_scene", "A_vlm"], [0.5, 0.5], spread(["A_agree", "A_scene", "A_vlm"]),
          "remove the detector", rem)
    score(j, ["A_object", "A_vlm"], [0.75, 0.25], spread(["A_object", "A_agree", "A_vlm"]),
          "remove the scene classifier", rem)
    score(j, ["A_object", "A_scene"], [0.75, 0.25], spread(["A_object", "A_agree", "A_scene"]),
          "remove the VLM", rem)
    out["ablation_what_can_go"] = rem

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
