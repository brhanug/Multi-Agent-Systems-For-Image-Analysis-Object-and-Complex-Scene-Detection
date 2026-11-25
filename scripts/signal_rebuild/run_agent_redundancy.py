#!/usr/bin/env python3
"""Is the four-agent formulation carrying four signals? (asked of the deployed design)

A reviewer's question: the agreement agent depends on the other two, and the object and VLM agents
look alike -- why are all four needed? This answers it from the stored scores rather than from the
design description.

  1 what each agent actually emits    its value set over the whole archive
  2 whether one is derived            A_agree against 1 - |A_obj - A_vlm|, the fallback rule in
                                      run_agent_comparison.py
  3 what the four add up to           the deployed uncertainty signal is the spread of the four; how
                                      many distinct values it can take
  4 whether the fourth agent pays     error prediction and scene decision on the 300 hand-annotated
                                      pictures, with and without each agent

The scene decision follows the thesis's Level-2 fusion (0.6 object + 0.2 scene + 0.2 VLM >= 0.5) and
an error is a picture where it disagrees with the human label, as in C4.

    python run_agent_redundancy.py [--verify]
"""
import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
TRUTH = BASE / "results/signal_rebuild/class_scores/human_class_truth_300.csv"
OUT = BASE / "results/signal_rebuild/agent_redundancy.json"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
SHORT = {"existing_pipeline_agent": "object", "agreement_agent": "agreement",
         "scene_agent": "scene", "vlm_agent": "vlm"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    d = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id")
    truth = pd.read_csv(TRUTH, keep_default_na=False).set_index("image_id")
    t = d.loc[truth.index]

    emits = {SHORT[c]: {"distinct_values_in_archive": int(d[c].nunique()),
                        "values": [round(float(v), 4) for v in sorted(d[c].unique())],
                        "source": str(d[c.replace("_agent", "") .replace("existing_pipeline", "object") + "_source"].iloc[0])}
             for c in A4}

    derived = np.clip(1.0 - (d.existing_pipeline_agent - d.vlm_agent).abs(), 0, 1)
    identity = {"rule_in_the_code": "A_agree = 1 - |A_obj - A_vlm| when the picture has no measured agreement score",
                "share_of_archive_matching_the_rule": round(float(np.isclose(d.agreement_agent, derived).mean()), 6),
                "largest_deviation": round(float((d.agreement_agent - derived).abs().max()), 9),
                "pictures_with_a_measured_agreement_score": int((d.agreement_source == "measured").sum())}

    spread4 = d[A4].std(axis=1)
    signal = {"definition": "population spread of the four agent scores, the deployed uncertainty signal",
              "distinct_values_over_12110_pictures": int(spread4.nunique()),
              "value_shares": {str(round(float(v), 4)): round(float(k) / len(d), 4)
                               for v, k in spread4.round(4).value_counts().sort_index().items()}}

    pairs = {f"{SHORT[x]}/{SHORT[y]}": {"identical": round(float((d[x] == d[y]).mean()), 4),
                                        "pearson_r": round(float(d[x].corr(d[y])), 4)}
             for x, y in combinations(A4, 2)}

    fused = 0.6 * t.existing_pipeline_agent + 0.2 * t.scene_agent + 0.2 * t.vlm_agent
    err = ((fused >= 0.5).astype(int) != truth["has_scene"].values).astype(int)
    y = truth["has_scene"].values
    drops = {"all four": A4}
    for c in A4:
        drops[f"without {SHORT[c]}"] = [x for x in A4 if x != c]
    error_prediction = {}
    for name, cols in drops.items():
        s = t[cols].std(axis=1)
        error_prediction[name] = {"auroc_at_predicting_errors": round(float(roc_auc_score(err, s)), 4),
                                  "distinct_values_on_the_300": int(s.nunique())}
    error_prediction["plain confidence (1 - |fused - 0.5|)"] = {
        "auroc_at_predicting_errors": round(float(roc_auc_score(err, 1 - (fused - 0.5).abs())), 4),
        "distinct_values_on_the_300": int((1 - (fused - 0.5).abs()).nunique())}

    decision = {}
    for name, cols in (("deployed fusion (object, scene, vlm)", None), ("with the agreement agent added", A4)):
        f = fused if cols is None else t[cols].mean(axis=1)
        decision[name] = {"auroc_against_the_human_scene_label": round(float(roc_auc_score(y, f)), 4),
                          "accuracy_at_0.5": round(float(((f >= 0.5).astype(int) == y).mean()), 4)}

    out = {"experiment": "Is the four-agent formulation carrying four signals?",
           "archive_pictures": int(len(d)), "evaluation_pictures": int(len(t)),
           "what_each_agent_emits": emits,
           "the_agreement_agent_is_derived": identity,
           "pairwise": pairs,
           "the_deployed_uncertainty_signal": signal,
           "error_prediction_on_the_300": error_prediction,
           "scene_decision_on_the_300": decision}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))

    print("what each level-1 agent emits, over the whole archive")
    for k, v in emits.items():
        print(f"  {k:10s} {v['distinct_values_in_archive']} distinct  {v['values']}  ({v['source']})")
    print(f"\nthe agreement agent equals 1 - |object - vlm| on "
          f"{identity['share_of_archive_matching_the_rule']:.2%} of pictures "
          f"(largest deviation {identity['largest_deviation']}); "
          f"{identity['pictures_with_a_measured_agreement_score']} pictures have a measured one")
    print("\npairwise, over the archive:")
    for k, v in pairs.items(): print(f"  {k:20s} identical {v['identical']:6.1%}   r {v['pearson_r']:+.3f}")
    print(f"\nthe deployed uncertainty signal takes {signal['distinct_values_over_12110_pictures']} distinct values")
    print("\npredicting the pipeline's errors on the 300:")
    for k, v in error_prediction.items():
        print(f"  {k:38s} AUROC {v['auroc_at_predicting_errors']:.3f}  ({v['distinct_values_on_the_300']} distinct values)")
    print("\nthe scene decision itself:")
    for k, v in decision.items():
        print(f"  {k:38s} AUROC {v['auroc_against_the_human_scene_label']:.3f}  accuracy {v['accuracy_at_0.5']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
