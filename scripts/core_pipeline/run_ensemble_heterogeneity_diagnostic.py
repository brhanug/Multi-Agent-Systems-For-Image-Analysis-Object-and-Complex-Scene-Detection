#!/usr/bin/env python3
"""
run_ensemble_heterogeneity_diagnostic.py
----------------------------------------
A pre-flight check for disagreement-based uncertainty on a new archive.

MOTIVATION. The Europeana transfer pilot showed that applying this pipeline
unchanged to a new collection can fail in a way that is invisible to the usual
metrics: not because the uncertainty signal is wrong, but because the ensemble
stops being an ensemble. If the Tier 1 agents collapse onto a single varying
signal on the target domain, then any "disagreement" computed over them is a
re-expression of one number, and its ROC-AUC is an artifact rather than a
measurement. This script measures that precondition BEFORE any labelling.

WHAT NOT TO USE. Three intuitive-looking indicators were tested and rejected
because they do NOT separate the working case from the failing one -- all three
hold just as strongly on Colibri, where the method demonstrably works:

  - "an agent takes only a couple of distinct values"  (VLM: 2 on Europeana,
    but also 2 across all 12,110 Colibri images)
  - "the fusion score correlates with one agent"       (r=0.993 Europeana,
    r=0.980 Colibri -- the 0.6 object weight makes fusion object-dominated by
    construction, everywhere)
  - "the scene agent is flat"                          (Europeana actually has
    MORE distinct scene values than Colibri: 22 vs 2)

Discreteness is not degeneracy. What matters is how much *variance* each agent
contributes, and how many independent directions the ensemble spans.

THE MEASURE. Effective rank of the agent-score covariance matrix: the
perplexity of its normalised eigenvalue spectrum,

    eff_rank = exp( -sum_i p_i log p_i ),   p_i = lambda_i / sum_j lambda_j

which answers "how many independent varying signals does this ensemble actually
carry here?" It is 4 for four independent agents and 1 if they collapse to one.
A near-constant agent contributes ~0 eigenvalue and correctly lowers the score.

CALIBRATION (the control condition that makes this meaningful):
    Colibri C0, n=12,110, method works        eff_rank = 2.50
    Europeana C5, n=24, method demonstrably    eff_rank = 1.37
      collapsed                                 [95% CI 1.06-1.61]

The Europeana interval does not approach the Colibri value, so the two regimes
are cleanly separated. The default threshold below sits between them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")

# Colibri reference value, measured on the full deployment corpus.
REFERENCE_EFF_RANK = 2.50
# Midpoint-ish between the working regime (2.50) and the collapsed one (1.37),
# above the upper bootstrap bound of the collapsed case (1.61).
DEFAULT_THRESHOLD = 2.0


def effective_rank(X: np.ndarray) -> float:
    """Perplexity of the normalised eigenvalue spectrum of cov(X)."""
    C = np.cov(X, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    total = ev.sum()
    if total <= 0:
        return 1.0
    p = ev / total
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def diagnose(df: pd.DataFrame, label: str, threshold: float = DEFAULT_THRESHOLD,
             n_boot: int = 2000, seed: int = 0) -> dict:
    d = df.copy()
    # Agreement is derived, not a separate model (Section: canonical SAA, Level 1)
    if "agreement_agent" not in d.columns:
        d["agreement_agent"] = 1 - (d["existing_pipeline_agent"] - d["vlm_agent"]).abs()
    # NOTE on conventions: U_SAA itself is the per-image SAMPLE std (ddof=1) over the
    # four agent columns, matching every script in this project that produces a cited
    # number. The effective rank below is a different quantity -- it is computed from
    # the ACROSS-image covariance matrix, where np.cov's default ddof=1 already applies
    # and the eigenvalue spectrum is scale-invariant, so the normalisation choice does
    # not affect the reported value.

    agents = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    X = d[agents].astype(float).values
    er = effective_rank(X)

    rng = np.random.default_rng(seed)
    boot = [effective_rank(X[rng.choice(len(X), len(X), replace=True)])
            for _ in range(n_boot)] if len(X) > 4 else []
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
        if boot else (float("nan"), float("nan"))

    per_agent = {a: {"std": round(float(d[a].std()), 4),
                     "n_unique": int(d[a].nunique())} for a in agents}
    usable = er >= threshold

    result = {
        "cohort": label,
        "n": int(len(d)),
        "effective_rank": round(er, 3),
        "effective_rank_ci95": [round(lo, 3), round(hi, 3)] if boot else None,
        "threshold": threshold,
        "colibri_reference": REFERENCE_EFF_RANK,
        "per_agent": per_agent,
        "disagreement_uncertainty_measurable": bool(usable),
        "verdict": (
            "PASS -- the ensemble spans enough independent directions here for "
            "cross-modal disagreement to be a meaningful signal."
            if usable else
            "FAIL -- the ensemble has collapsed toward a single varying signal on "
            "this data. Disagreement computed over it re-expresses one number, so "
            "any error-detection AUROC would be an artifact. Recalibrate or "
            "substitute agents before claiming transfer."
        ),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--scores-csv", required=True,
                    help="CSV with columns existing_pipeline_agent, scene_agent, "
                         "vlm_agent (agreement_agent derived if absent)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    res = diagnose(df, args.label or Path(args.scores_csv).stem, args.threshold)

    print(f"\n{res['cohort']}  (n={res['n']})")
    print(f"  effective rank      : {res['effective_rank']} of 4"
          + (f"   95% CI {res['effective_rank_ci95']}" if res['effective_rank_ci95'] else ""))
    print(f"  Colibri reference   : {REFERENCE_EFF_RANK}     threshold: {args.threshold}")
    for a, v in res["per_agent"].items():
        print(f"    {a:26s} std={v['std']:.4f}  n_unique={v['n_unique']}")
    print(f"\n  {res['verdict']}\n")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out_json, "w"), indent=2)
        print(f"  saved -> {args.out_json}")


if __name__ == "__main__":
    main()
