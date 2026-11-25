#!/usr/bin/env python3
"""The scene pipeline re-run end to end with every Tier-1 agent doing its specified job.

The deployed Tier-1 vector does not contain what its description says. A_object is a VQA object
count saturated at three (the spatial detector returned no boxes at all), A_vlm is the indicator
1[A_object > 0], A_scene is a softmax over five scene prompts -- a forced choice that cannot express
absence -- and A_agree is 1 - |A_object - A_vlm|, algebra over two columns. Three measured signals
with 4, 2 and 2 values realise 8 joint states, so the whole vector carries at most three bits and any
function of it takes at most eight values.

This run replaces each agent with its specified output, keeping scene presence as the target:

  spatial grounding   the rebuilt detector's per-class confidences over the 8 frozen classes
  zero-shot scene     SigLIP's sigmoid over the 5 scene prompts, its calibrated native channel,
                      in place of the softmax that scores below chance at scene presence
  visual reasoning    LLaVA's per-class P(yes) read from first-token logprobs, over the 8 classes
  agreement           dropped: it was never measured on any picture

Commensurability. The deployed fusion averages a count, a flag and a binary answer, so its spread
partly measures how those scales were normalised rather than models disagreeing. Here each agent
keeps its native vector and is mapped to a probability of scene presence by a logistic fit on the
TRAIN split alone. The three agents then estimate the same quantity on the same scale, which is what
makes both the weighted mean and the spread meaningful. No agent sees another's features.

Split discipline: mappings fitted on train (749), the conformal threshold calibrated on calibration
(252), everything reported on test (249). The deployed pipeline is scored on the same test pictures.

    python run_corrected_tier1_end_to_end.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
CS = SR / "class_scores"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = SR / "corrected_tier1_end_to_end.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
WEIGHTS = {"object": 0.6, "scene": 0.2, "vlm": 0.2}   # the deployed Level-2 weights, unchanged
GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]


def native_vectors():
    """Each agent's own output, in its own form, for every picture in the archive.

    Built by tier1_signals.load, the single construction point, so the guards run before any fusion
    reads these: measured, covered, continuous, independent. The old detector's column is read
    separately, only to isolate how much of the gain is the rebuilt detector's."""
    import tier1_signals

    sig = tier1_signals.load("corrected")          # raises if any guard fails
    abs_sig = tier1_signals.load("corrected", absence=True)
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    old_det = np.nan_to_num(m["raw"][:, ms.index("detector")][:, [mc.index(c) for c in CLASSES]].astype(np.float64))
    return (sig["index"], {"object": sig["object"], "scene": sig["scene"], "vlm": sig["vlm"]}, old_det,
            abs_sig["scene"])


def to_probability(V, rows_fit, y_fit, rows_all):
    """One agent's native vector -> a probability of scene presence, fitted on the train split only."""
    mdl = LogisticRegression(max_iter=2000, C=1.0).fit(V[rows_fit], y_fit)
    return mdl.predict_proba(V[rows_all])[:, 1]


def crc(U_cal, E_cal, U_te, E_te, alpha):
    n = len(U_cal)
    risks = np.array([np.mean(E_cal * (U_cal < g)) for g in GRID])
    ok = (n / (n + 1.0)) * risks + (1.0 / (n + 1.0)) <= alpha
    lam = float(np.max(GRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
    auto = U_te < lam
    caught = int(E_te[~auto].sum()); tot = int(E_te.sum())
    return {"alpha": alpha, "lambda": round(lam, 4),
            "test_empirical_risk": round(float(np.mean(E_te * auto)), 4),
            "bound_satisfied": bool(float(np.mean(E_te * auto)) <= alpha),
            "auto_approval_pct": round(100 * float(auto.mean()), 2),
            "error_recall_pct": round(100 * caught / tot, 2) if tot else None}


def report(saa, unc, y, err, tag):
    return {"which": tag,
            "scene_ranking_auroc": round(float(roc_auc_score(y, saa)), 4),
            "test_error_rate": round(float(err.mean()), 4),
            "uncertainty_auroc_at_predicting_errors": round(float(roc_auc_score(err, unc)), 4)
            if 0 < err.sum() < len(err) else None,
            "plain_confidence_auroc": round(float(roc_auc_score(err, 1 - np.abs(saa - 0.5))), 4)
            if 0 < err.sum() < len(err) else None,
            "distinct_uncertainty_values": int(len(np.unique(np.round(unc, 9))))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    idx, nat, old_det, scene_abs = native_vectors()
    sp = pd.read_csv(SPLIT)
    sp = sp[sp.ppn_id.isin(idx)].copy()
    sp["row"] = sp.ppn_id.map(idx)
    part = {k: sp[sp.split == k] for k in ("train", "calibration", "test")}
    rows = {k: v.row.values for k, v in part.items()}
    ys = {k: v.has_scene.values for k, v in part.items()}
    allrows = np.concatenate([rows["train"], rows["calibration"], rows["test"]])

    out = {"experiment": "The scene pipeline end to end with each Tier-1 agent doing its specified job",
           "target": "scene presence, the deployed target, unchanged",
           "agreement_agent": "dropped -- 1 - |A_object - A_vlm| on 12,110 of 12,110 pictures, never measured",
           "commensurability": "each agent's native vector mapped to P(scene) by a logistic fit on train only",
           "split": {k: int(len(v)) for k, v in rows.items()},
           "level2_weights": WEIGHTS, "decision_threshold": 0.5}

    variants = {"corrected (rebuilt detector)": nat,
                "corrected (deployed detector)": {**nat, "object": old_det},
                "corrected + absence router": {**nat, "scene": scene_abs}}
    out["agents"] = {}; out["variants"] = {}
    for tag, V in variants.items():
        P = {}
        for name, M in V.items():
            full = to_probability(M, rows["train"], ys["train"], allrows)
            P[name] = dict(zip(allrows, full))
        pos = {k: np.array([[P[n][r] for n in ("object", "scene", "vlm")] for r in rows[k]])
               for k in rows}
        if tag.startswith("corrected (rebuilt"):
            out["agents"] = {n: {"native_dimension": int(V[n].shape[1]),
                                 "auroc_on_test_alone": round(float(roc_auc_score(
                                     ys["test"], pos["test"][:, i])), 4)}
                             for i, n in enumerate(("object", "scene", "vlm"))}
        w = np.array([WEIGHTS["object"], WEIGHTS["scene"], WEIGHTS["vlm"]])
        saa = {k: pos[k] @ w for k in rows}
        unc = {k: pos[k].std(axis=1) for k in rows}
        err = {k: ((saa[k] >= 0.5).astype(int) != ys[k]).astype(int) for k in rows}
        r = report(saa["test"], unc["test"], ys["test"], err["test"], tag)
        r["conformal_risk_control"] = [crc(unc["calibration"], err["calibration"],
                                           unc["test"], err["test"], al) for al in ALPHAS]
        # AR9: the same guarantee, scored by confidence instead of disagreement. Confidence is the
        # better error predictor in every corrected variant, and the triage score is exactly where
        # that matters -- it decides which pictures a person never sees.
        conf = {k: 1 - np.abs(saa[k] - 0.5) for k in rows}
        r["conformal_risk_control_on_confidence"] = [crc(conf["calibration"], err["calibration"],
                                                         conf["test"], err["test"], al) for al in ALPHAS]
        out["variants"][tag] = r

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id")
    A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    dsaa, dunc, derr = {}, {}, {}
    for k, v in part.items():
        t = ag.loc[v.ppn_id]
        dsaa[k] = (0.6 * t.existing_pipeline_agent + 0.2 * t.scene_agent + 0.2 * t.vlm_agent).values
        dunc[k] = t[A4].values.std(axis=1)
        derr[k] = ((dsaa[k] >= 0.5).astype(int) != ys[k]).astype(int)
    r = report(dsaa["test"], dunc["test"], ys["test"], derr["test"], "deployed")
    r["conformal_risk_control"] = [crc(dunc["calibration"], derr["calibration"],
                                       dunc["test"], derr["test"], al) for al in ALPHAS]
    dconf = {k: 1 - np.abs(dsaa[k] - 0.5) for k in rows}
    r["conformal_risk_control_on_confidence"] = [crc(dconf["calibration"], derr["calibration"],
                                                     dconf["test"], derr["test"], al) for al in ALPHAS]
    cf = ag.loc[part["test"].ppn_id, "comparison_fusion_score"].values
    r["note_on_an_earlier_figure"] = {
        "test_error_rate_under_comparison_fusion_score": round(
            float((((cf >= 0.5).astype(int) != ys["test"]).astype(int)).mean()), 4),
        "why": ("end_to_end_rebuilt.json reports a deployed test error of 0.2249. It scores the "
                "deployed arm with comparison_fusion_score, the four-agent mean weighted by "
                "config.yaml, while scoring the rebuilt arm with the Level-2 SAA. Both arms here "
                "use the Level-2 fusion the thesis defines, 0.6 object + 0.2 scene + 0.2 VLM, so "
                "the deployed error is higher and the comparison is like for like.")}
    out["variants"]["deployed"] = r

    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))

    print(f"split: train {len(rows['train'])}, calibration {len(rows['calibration'])}, test {len(rows['test'])}")
    print("\neach agent alone on test, after its own mapping to P(scene):")
    for n, v in out["agents"].items():
        print(f"  {n:8s} native vector of {v['native_dimension']:2d}   AUROC {v['auroc_on_test_alone']:.3f}")
    print(f"\n{'':32s} {'scene AUROC':>11s} {'error':>7s} {'U at errors':>11s} {'confidence':>10s} {'U values':>9s}")
    for tag, v in out["variants"].items():
        print(f"  {tag:30s} {v['scene_ranking_auroc']:11.3f} {v['test_error_rate']:7.3f} "
              f"{(v['uncertainty_auroc_at_predicting_errors'] or 0):11.3f} {(v['plain_confidence_auroc'] or 0):10.3f} "
              f"{v['distinct_uncertainty_values']:9d}")
    for key, title in (("conformal_risk_control", "triage scored by disagreement"),
                       ("conformal_risk_control_on_confidence", "triage scored by confidence")):
        print(f"\nconformal risk control, {title} (auto-approval % at each target):")
        for tag, v in out["variants"].items():
            if key not in v: continue
            cells = "  ".join(f"a={c['alpha']}: {c['auto_approval_pct']:5.1f}%{'' if c['bound_satisfied'] else '!'}"
                              for c in v[key])
            print(f"  {tag:30s} {cells}")


if __name__ == "__main__":
    raise SystemExit(main())
