"""
Quantify how much the headline error-detection figure depends on the uncertainty
signal sharing inputs with the label that defines "error".

The deployed error label thresholds a weighted mean of the same agent scores from
which U_SAA is computed. This measures the same task with signals that do NOT
feed the label, rebuilt from the raw model outputs, so the gap between the two is
an estimate of how much the headline number owes to that shared construction.

The comparison is deliberately conservative in one direction and should be read
with that in mind: the rebuilt signals are predicting a label defined by a
different system's agents, which is a harder task than predicting a label they
help define. The gap is therefore an upper bound on the inflation, not a point
estimate of "the real number".

    python measure_error_label_circularity.py [--verify]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from run_noncircular_disagreement_test import load, scale, MODELS

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/error_label_circularity.json"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
A3 = ["existing_pipeline_agent", "scene_agent", "vlm_agent"]
SEED, N_BOOT = 0, 4000


def ci(y, s, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], s[i]))
    o = np.array(o)
    return {"auc": round(float(roc_auc_score(y, s)), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    full, d = load()
    tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
    cols = list(MODELS.values())
    _, tes = scale(tr, te, cols)

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = tes.merge(ag, left_on="ppn_id", right_on="image_id", how="inner", suffixes=("", "_dep"))
    fusion = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    err = ((fusion >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int).values

    out = {"experiment": "How much of the headline figure comes from the label sharing the signal's inputs",
           "n_test": int(len(j)), "n_errors": int(err.sum()),
           "error_label": "1[0.6*A_object + 0.2*A_scene + 0.2*A_vlm >= 0.5] != human has_scene",
           "signals_that_HELP_DEFINE_the_label": {
               "U_SAA (deployed, 4 terms)": ci(err, j[A4].std(axis=1, ddof=1).values),
               "1 - mean of the 3 fusion agents": ci(err, (1 - j[A3].mean(axis=1)).values)},
           "signals_that_do_NOT_feed_the_label": {
               "rebuilt disagreement (4 continuous, separate models)":
                   ci(err, tes[cols].std(axis=1, ddof=1).values),
               "rebuilt 1 - mean confidence": ci(err, (1 - tes[cols].mean(axis=1)).values)}}

    shared = max(v["auc"] for v in out["signals_that_HELP_DEFINE_the_label"].values())
    indep = max(v["auc"] for v in out["signals_that_do_NOT_feed_the_label"].values())
    out["gap"] = {"best_sharing_inputs_with_label": shared,
                  "best_not_feeding_label": indep,
                  "difference": round(shared - indep, 4),
                  "reading": "An upper bound on how much the headline owes to shared construction. "
                             "The rebuilt signals also face a harder task -- predicting a label "
                             "defined by a different system's agents -- so part of this gap is task "
                             "difficulty rather than circularity."}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
