"""
Which uncertainty formulation is actually defensible?

Three defects were established in the deployed definition:
  (i)   A_agree is derived from A_object and A_vlm, not measured;
  (ii)  A_vlm = 1[A_object > 0] on every image, so it carries no new information;
  (iii) consequently U_SAA is an 8-valued function of two free inputs.

The obvious repair -- drop the derived term -- does NOT fix this, because defect
(ii) survives it. This evaluates every candidate on the same corrected held-out
partition, reporting free-input count and resolution alongside performance so the
trade is visible.

    python run_corrected_formulation.py [--verify]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
REBUILT = BASE / "results/signal_rebuild/raw_agent_signals.csv"
SIG = BASE / "results/signal_rebuild/siglip_scene_full.jsonl"
OUT = BASE / "results/multi_agent/corrected_formulation.json"
O, A, S, V, C = ("existing_pipeline_agent", "agreement_agent", "scene_agent",
                 "vlm_agent", "document_agent")
SEED, N_BOOT = 0, 4000


def canonical(i):
    i = str(i)
    if i.startswith("PPN"):
        k = i.find("_")
        if k > 0:
            return i[:k] + "/" + i[k + 1:]
    return i


def ci(y, s, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], s[i]))
    o = np.array(o)
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)],
            "pr_auc": round(float(average_precision_score(y, s)), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    sp = pd.read_csv(SPLIT)
    d = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    d["fus"] = 0.6 * d[O] + 0.2 * d[S] + 0.2 * d[V]
    d["err"] = ((d.fus >= SAA_THRESHOLD).astype(int) != d.has_scene).astype(int)

    # rebuilt signals, joined on the same rows
    rb = pd.read_csv(REBUILT); rb["image_id"] = rb.image_id.map(canonical)
    sigm = {}
    with open(SIG) as f:
        for line in f:
            try:
                r = json.loads(line); sigm[canonical(r["image_id"])] = r["max_sigmoid"]
            except Exception:
                pass
    rb["siglip"] = rb.image_id.map(sigm)
    rb["ocrq"] = np.minimum(rb.ocr_chars / 80.0, 1.0) * (1 - rb.ocr_repetition)
    d = d.merge(rb[["image_id", "gd_conf_mean", "siglip", "vlm_gd_jaccard", "ocrq"]],
                left_on="ppn_id", right_on="image_id", how="left", suffixes=("", "_rb"))
    te = d[d.split == "test"].reset_index(drop=True)

    def mm(col):  # min-max on the test rows, so the four are comparable
        v = te[col].astype(float)
        return ((v - v.min()) / max(v.max() - v.min(), 1e-9)).values

    cands = {
        "DEPLOYED  std(Object, Agreement, Scene, VLM)":
            (te[[O, A, S, V]].std(axis=1, ddof=1).values, 2,
             "A_agree derived; A_vlm = 1[A_object>0]"),
        "REPAIR 1  std(Object, Scene, VLM)  [drop derived term]":
            (te[[O, S, V]].std(axis=1, ddof=1).values, 2,
             "still 2 free inputs -- A_vlm remains a function of A_object"),
        "REPAIR 2  std(Object, Scene, OCR)  [drop derived AND redundant]":
            (te[[O, S, C]].std(axis=1, ddof=1).values, 3,
             "OCR is the only deployed signal not tied to A_object"),
        "REPAIR 3  std over the four rebuilt signals":
            (np.column_stack([mm("gd_conf_mean"), mm("siglip"),
                              mm("vlm_gd_jaccard"), mm("ocrq")]).std(axis=1, ddof=1), 4,
             "re-derived from raw model outputs; four separate models"),
        "BASELINE  1 - mean(Object, Scene, VLM)  [confidence]":
            ((1 - te[[O, S, V]].mean(axis=1)).values, 2, "no disagreement component"),
    }

    res = {"experiment": "Which uncertainty formulation is defensible?",
           "cohort": "corrected group-level held-out test",
           "n": int(len(te)), "n_errors": int(te.err.sum()),
           "note": "The error label thresholds a fusion of the deployed agents, so formulations "
                   "built from those same agents are advantaged here. Repair 3, built from "
                   "separately re-derived signals, is the only row not favoured by that.",
           "candidates": {}}
    for name, (sig, free, why) in cands.items():
        res["candidates"][name] = ci(te.err.values, sig) | {
            "free_inputs": free,
            "distinct_values_on_test": int(len(np.unique(np.round(sig, 6)))),
            "why": why}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(res, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
