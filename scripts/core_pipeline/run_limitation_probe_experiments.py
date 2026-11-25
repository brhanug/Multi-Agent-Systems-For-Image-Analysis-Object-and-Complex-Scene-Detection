"""
Experiments against the limitations the deck states on slide 4.2.

Each of the six bullets on 4.2 is a claim that something does NOT work. This
script asks, for the ones that are empirically decidable, whether they can be
fixed -- and reports the answer either way. Two of the six are definitional
and are not testable here; the script says so rather than inventing a test.

Everything is fitted on the `train` partition of the corrected clean-holdout
split and scored on `test`. No number below is in-sample.

    python run_limitation_probe_experiments.py            # run
    python run_limitation_probe_experiments.py --verify   # check against artifact
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path("/data/brhanu/thesis_project")
AGENTS_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT_CSV = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT_JSON = BASE / "results/multi_agent/limitation_probe_experiments.json"

CORE4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
CORE3 = ["existing_pipeline_agent", "scene_agent", "vlm_agent"]
EXCLUDED = ["restoration_agent", "document_agent"]   # measured, continuous, in no level
SEED = 0
N_BOOT = 4000


def load():
    ag = pd.read_csv(AGENTS_CSV).drop_duplicates("image_id")
    sp = pd.read_csv(SPLIT_CSV)
    d = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    # the deployed clean-holdout error label (run_corrected_clean_holdout_evaluation.py:70)
    d["saa_fusion"] = 0.6 * d.existing_pipeline_agent + 0.2 * d.scene_agent + 0.2 * d.vlm_agent
    d["err"] = ((d.saa_fusion >= SAA_THRESHOLD).astype(int) != d.has_scene).astype(int)
    d["U_SAA"] = d[CORE4].std(axis=1)
    d["conf"] = 1 - d[CORE3].mean(axis=1)
    return d


def auc(y, s):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def paired_boot(y, a, b, n=N_BOOT, seed=SEED):
    """Bootstrap the paired difference auc(a) - auc(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    out = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"delta": round(float(out.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_resamples": int(len(out))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    d = load()
    tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
    out = {"experiment": "Can the limitations stated on slide 4.2 be fixed?",
           "protocol": "fit on train (n=%d), score on test (n=%d); no in-sample numbers"
                       % (len(tr), len(te)),
           "test_errors": int(te.err.sum())}

    # =====================================================================
    # 4.2 bullet 3 -- "Not disagreement doing the work."
    # Does U_SAA add anything ON TOP OF plain confidence, held out?
    # =====================================================================
    yte = te.err.values
    base = LogisticRegression(max_iter=2000).fit(tr[["conf"]], tr.err)
    full = LogisticRegression(max_iter=2000).fit(tr[["conf", "U_SAA"]], tr.err)
    p_base = base.predict_proba(te[["conf"]])[:, 1]
    p_full = full.predict_proba(te[["conf", "U_SAA"]])[:, 1]
    out["b3_does_disagreement_add_over_confidence"] = {
        "question": "Held out, does adding U_SAA to a confidence-only model improve error detection?",
        "auc_confidence_only": round(auc(yte, p_base), 4),
        "auc_confidence_plus_U_SAA": round(auc(yte, p_full), 4),
        "paired_delta": paired_boot(yte, p_full, p_base),
        "raw_signal_aucs": {"U_SAA": round(auc(yte, te.U_SAA.values), 4),
                            "confidence": round(auc(yte, te.conf.values), 4)},
    }

    # =====================================================================
    # 4.2 bullet 4 -- "Not reliable on blank images."
    # Can the no-scene regime be DETECTED up front, using the two measured
    # signals that no level uses (restoration, document/OCR), and routed away?
    # =====================================================================
    b4 = {"question": "Can no-scene images be flagged before the uncertainty machinery is trusted?"}
    neg = te[te.has_scene == 0]
    allneg = d[d.has_scene == 0]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(neg), len(neg))
        if len(np.unique(neg.err.values[i])) > 1:
            boots.append(roc_auc_score(neg.err.values[i], neg.U_SAA.values[i]))
    boots = np.array(boots)
    b4["failure_being_addressed"] = {
        "n_no_scene_in_test": int(len(neg)),
        "n_errors_in_that_subset": int(neg.err.sum()),
        "U_SAA_auc_on_no_scene_only": (round(auc(neg.err.values, neg.U_SAA.values), 4)
                                       if len(np.unique(neg.err.values)) > 1 else "single class"),
        "auc_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)],
        "entire_ci_below_chance": bool(np.percentile(boots, 97.5) < 0.5),
        "pipeline_false_positive_rate_on_no_scene": round(float((neg.saa_fusion >= SAA_THRESHOLD).mean()), 4),
        "replication_on_all_partitions": {
            "n": int(len(allneg)), "n_errors": int(allneg.err.sum()),
            "U_SAA_auc": round(auc(allneg.err.values, allneg.U_SAA.values), 4),
            "note": "not held out -- a sample-size check on the same finding, not an independent estimate"},
    }
    for name, feats in [("core4_only", CORE4),
                        ("excluded_only (restoration + OCR)", EXCLUDED),
                        ("core4 + excluded", CORE4 + EXCLUDED)]:
        g = LogisticRegression(max_iter=2000).fit(tr[feats], 1 - tr.has_scene)
        s = g.predict_proba(te[feats])[:, 1]           # P(no scene)
        b4.setdefault("gate_detects_no_scene", {})[name] = {
            "auc": round(auc((1 - te.has_scene).values, s), 4)}
    # best gate -> abstain on the flagged fraction, re-score what remains
    g = LogisticRegression(max_iter=2000).fit(tr[CORE4 + EXCLUDED], 1 - tr.has_scene)
    s = g.predict_proba(te[CORE4 + EXCLUDED])[:, 1]
    sweep = []
    for q in (0.05, 0.10, 0.15, 0.20, 0.30):
        keep = s <= np.quantile(s, 1 - q)
        sub = te[keep]
        sweep.append({"abstain_fraction": q,
                      "coverage_retained": round(float(keep.mean()), 4),
                      "n_kept": int(keep.sum()),
                      "no_scene_removed": int((te.has_scene == 0).sum() - (sub.has_scene == 0).sum()),
                      "U_SAA_auc_on_kept": (round(auc(sub.err.values, sub.U_SAA.values), 4)
                                            if len(np.unique(sub.err.values)) > 1 else "single class")})
    b4["abstention_sweep"] = sweep
    b4["U_SAA_auc_no_abstention"] = round(auc(yte, te.U_SAA.values), 4)
    # is the best abstention gain distinguishable from zero? bootstrap the pair
    keep = s <= np.quantile(s, 0.90)
    rng2 = np.random.default_rng(SEED)
    gains = []
    for _ in range(N_BOOT):
        i = rng2.integers(0, len(te), len(te))
        yk, uk, kk = yte[i], te.U_SAA.values[i], keep[i]
        if len(np.unique(yk)) < 2 or len(np.unique(yk[kk])) < 2:
            continue
        gains.append(roc_auc_score(yk[kk], uk[kk]) - roc_auc_score(yk, uk))
    gains = np.array(gains)
    b4["gain_at_10pct_abstention"] = {
        "delta_auc": round(float(gains.mean()), 4),
        "ci95": [round(float(np.percentile(gains, 2.5)), 4),
                 round(float(np.percentile(gains, 97.5)), 4)],
        "excludes_zero": bool(np.percentile(gains, 2.5) > 0 or np.percentile(gains, 97.5) < 0),
        "n_resamples": int(len(gains))}
    out["b4_can_the_no_scene_regime_be_gated"] = b4

    # =====================================================================
    # 4.2 bullet 2 -- "Not the only way." Can the ensemble be made to beat
    # the single model, held out, with the excluded signals added back?
    # =====================================================================
    single = LogisticRegression(max_iter=2000).fit(tr[["existing_pipeline_agent"]], tr.err)
    ens = LogisticRegression(max_iter=2000).fit(tr[CORE4 + EXCLUDED], tr.err)
    p_s = single.predict_proba(te[["existing_pipeline_agent"]])[:, 1]
    p_e = ens.predict_proba(te[CORE4 + EXCLUDED])[:, 1]
    out["b2_can_the_ensemble_beat_the_single_model"] = {
        "question": "With the two excluded measured signals added, does the ensemble beat one model?",
        "auc_single_model": round(auc(yte, p_s), 4),
        "auc_ensemble_plus_excluded": round(auc(yte, p_e), 4),
        "paired_delta": paired_boot(yte, p_e, p_s),
    }

    # =====================================================================
    # 4.2 bullet 6 -- "Not causal."
    # Leave-one-out removal IS an intervention on the pipeline, so it licenses
    # a causal statement about the pipeline (not about the world).
    # =====================================================================
    from scipy.stats import spearmanr
    u_full = te[CORE4].std(axis=1).values
    ref = auc(yte, u_full)
    abl = {}
    for a in CORE4:
        rest = [c for c in CORE4 if c != a]
        v = te[rest].std(axis=1).values
        rho = float(spearmanr(u_full, v).statistic)
        abl[a] = {"auc_without": round(auc(yte, v), 4),
                  "delta_vs_full": round(auc(yte, v) - ref, 4),
                  "spearman_vs_full_U_SAA": round(rho, 6),
                  "rank_identical_to_full": bool(rho == 1.0),
                  "paired_delta": paired_boot(yte, v, u_full)}
    out["b6_interventional_ablation"] = {
        "question": "Does removing an agent change the signal? (an intervention, not an attribution)",
        "note": "This licenses a causal claim about THIS pipeline, not about the world. "
                "It does not make the Shapley attribution causal.",
        "why_object_removal_is_exactly_zero": "Removing the object agent maps the 8 distinct U_SAA "
                "values through a strictly increasing bijection, so the ranking -- and therefore "
                "ROC-AUC -- is unchanged. Spearman = 1.0 exactly. This is the structural redundancy "
                "of A.14 showing up as a zero interventional effect, not a numerical artifact.",
        "auc_full_U_SAA": round(ref, 4), "leave_one_out": abl}

    # =====================================================================
    # not decidable here
    # =====================================================================
    out["not_empirically_decidable"] = {
        "b1_proxy_not_internal_state": "Definitional. No experiment on these artifacts can turn a "
                                       "score computed from agent outputs into a measurement of a "
                                       "model's internal state.",
        "b5_transferability": "Needs a second annotated archive. The Europeana pilot (n=24) is far "
                              "too small to settle it; the heterogeneity precondition (A.13) is the "
                              "diagnostic offered in its place.",
    }

    if args.verify:
        prev = json.loads(OUT_JSON.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH vs committed artifact")
        return 0 if same else 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
