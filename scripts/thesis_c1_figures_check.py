#!/usr/bin/env python3
"""Hold the thesis's in-sample figures on the 801 expert-reviewed pictures (C1) to their result files.

These figures come from the deployed-design analyses in results/multi_agent/, which neither
thesis_rerun_figures.py nor thesis_v4_chapter6_check.py covers. On 3 November 2026 they were found
to join expert labels to agent scores on the page number alone (548 of 801 pictures matched to
another book); the scripts now join on book and page (scripts/core_pipeline/picture_ids.py) and were
rerun. Each entry is a piece of the text with ⟨⟩ where a figure sits, its result file and key, and
how it is printed; the filled text must occur exactly once in the thesis. Nothing is rewritten.

    python thesis_c1_figures_check.py
"""
import json, sys
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
MA, SR = BASE / "results/multi_agent", BASE / "results/signal_rebuild"
FILES = [BASE / "presentation/thesis_v3_updated.tex", BASE / "latex_assets/failure_gallery.tex"]
M = "⟨⟩"
_cache = {}


def val(spec):
    f, keys = spec.split(":", 1)
    root = SR if f in ("c1_platt_check.json", "example_pictures.json", "per_batch_risk.json", "external_audit.json") else MA
    d = _cache.setdefault(f, json.loads((root / f).read_text()))
    for k in keys.split("|"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


f3, f4 = (lambda x: f"{x:.3f}"), (lambda x: f"{x:.4f}")
pct1v = lambda x: f"{x:.1f}"                      # a value already in percent
D = "uncertainty_signal_decomposition.json:baseline_comparison|C1_in_sample|signals|"
U, S3, PC, SM = (D + "U_SAA as deployed (4 signals, incl. derived Agreement)", D + "sigma over the 3 genuine agents (derived removed)",
                 D + "1 - mean(3 genuine agents)  [pure confidence]", D + "1 - A_object  [single-model confidence]")
H = "uncertainty_signal_decomposition.json:baseline_comparison|clean_holdout_test|signals|1 - mean(3 genuine agents)  [pure confidence]|paired_diff_vs_deployed_U_SAA|"
CRC = "exp29_crc_results.json:results|2|"
AW = "exp20_error_awlf_results.json:cv_results|"
SH = "exp30_shapley_weighted_fusion_results.json:metrics|"
EX6 = "detailed_viva_experiments_report.json:exp6_uncertainty_baseline_comparison|"
PL = "c1_platt_check.json:ece_test_half|"

pct100, intg = (lambda x: f"{100 * x:.1f}"), (lambda x: str(int(x)))

ENTRIES = [
    # --- accuracy of the deployed design's automatic approvals, Task E (added 7 Nov 2026)
    ('the scene rule of Section~\\ref{sec:archive_check}: $⟨⟩$ of them', 'external_audit.json:original_design_on_task_E|wrong', intg),
    ('$52$ of them ($⟨⟩\\%$, $95\\%$ interval $[16.4', 'external_audit.json:original_design_on_task_E|wrong_share', pct100),
    # --- Agent 2 precision on all 127 candidates (added 5 Nov 2026)
    ('the $116$ unambiguous cases) are hallucinations, $⟨⟩$ are YOLO', 'agent2_contradiction_precision_real.json:n_present_yolo_miss_not_hallucination', intg),
    ('without those, $22$ of $92$ ($⟨⟩\\%$) are hallucinations', 'agent2_contradiction_precision_real.json:without_the_24_words_on_one_emblem|precision_strict', pct100),
    ('were hallucinations ($⟨⟩\\%$ without $24$ words', 'agent2_contradiction_precision_real.json:without_the_24_words_on_one_emblem|precision_strict', pct100),
    ('stand alone: the contradiction checker $⟨⟩\\%$ on the first prototype', 'agent2_contradiction_precision_real.json:precision_as_hallucination_detector_strict', pct100),
    # --- the deployed score on C1, and what it measures
    ("\\textbf{SAA disagreement proxy}: \\textbf{ROC-AUC = ⟨⟩}", U + "|roc_auc", f4),
    ("Temperature-Scaled Entropy Baseline: $\\text{ROC-AUC} = ⟨⟩$", EX6 + "1|ROC-AUC", f4),
    ("drops in-sample error-detection ROC-AUC from $\\mathbf{⟨⟩}$ to", U + "|roc_auc", f4),
    ("to $\\mathbf{⟨⟩}$ ($95\\%$ CI $[0.711, 0.782]$)", S3 + "|roc_auc", f4),
    ("incl.\\ derived) & $\\mathbf{⟨⟩}$ & $\\mathbf{0.8600}$", U + "|roc_auc", f4),
    ("(single-model confidence) & $⟨⟩$ & $0.8722$", SM + "|roc_auc", f4),
    ("(pure confidence) & $⟨⟩$ & $0.8799$", PC + "|roc_auc", f4),
    ("(derived term removed) & $⟨⟩$ & $0.8340$", S3 + "|roc_auc", f4),
    ("pure confidence minus deployed $U_{\\text{SAA}}$ is $⟨⟩$ ($95\\%$ CI $[-0.0034, +0.0004]$)", PC + "|paired_diff_vs_deployed_U_SAA|mean", f4),
    ("on the corrected held-out partition it is $+⟨⟩$ ($95\\%$ CI $[+0.0009, +0.0410]$)", H + "mean", f4),
    ("on discrimination ($⟨⟩$ vs.\\ $0.9767$)", U + "|roc_auc", f4),
    ("Object signal alone ($1-A^{\\text{object}}$) & ⟨⟩", SM + "|roc_auc", f4),
    ("Scene signal alone (temperature-scaled entropy) & ⟨⟩", EX6 + "1|ROC-AUC", f4),
    ("fusion core ($U_{\\text{SAA}}$) & \\textbf{⟨⟩}", U + "|roc_auc", f4),
    ("a single model's own confidence ties it ($⟨⟩$ against $0.9791$ on $C_1$)", SM + "|roc_auc", f4),
    ("the in-sample ROC-AUC drops to $⟨⟩$, while a plain confidence score", S3 + "|roc_auc", f4),
    ("with no disagreement information scores $⟨⟩$ (Section", PC + "|roc_auc", f4),
    ("drops in-sample ROC-AUC from $0.9791$ to $⟨⟩$, and that", S3 + "|roc_auc", f4),
    ("but $⟨⟩$ under the three-agent definition", "formulation_sensitivity.json:C_agent_ablation_on_error_detection|C1_in_sample|full_4_term|roc_auc", f4),
    ("on $C_1$ it affects $⟨⟩$ of\n$796$", "formulation_sensitivity.json:D_structural_blind_spot_of_the_fusion|C1_in_sample|n_scene_positive_with_A_object_zero", str),
    # --- Europeana pilot: the Colibri C1 baseline
    ("\\textbf{SAA fusion score}   & \\textbf{0.332} & \\textbf{⟨⟩}", "colibri_c1_baseline.json:mean_fusion_score", f3),
    ("(canonical, cross-agent std) & 0.361 & ⟨⟩", "colibri_c1_baseline.json:mean_U_SAA", f3),
    ("lower than the Colibri baseline ($⟨⟩$)", "colibri_c1_baseline.json:mean_fusion_score", f3),
    # --- not-claimed list: the per-batch risk promise
    ("exceeds the level in $⟨⟩\\%$ of cases; calibrating", "per_batch_risk.json:levels|0.05|share_over_0.05", lambda x: f"{100 * x:.1f}"),
    ("while still approving $⟨⟩\\%$", "per_batch_risk.json:levels|0.025|mean_approved", lambda x: f"{100 * x:.1f}"),
    # --- Platt scaling on C1
    ("the calibrated probability reaches an ECE of $⟨⟩$ ($95\\%$ range", PL + "platt|mean", f4),
    ("against $⟨⟩$ before calibration", PL + "raw|mean", f3),
    ("always predicting the base rate reaches the same $⟨⟩$", PL + "base_rate|mean", f4),
    # --- in-sample conformal risk control
    ("automatically approves \\textbf{⟨⟩\\% of the collection}", CRC + "test_auto_approval_rate", pct1v),   # figure removed 21 Nov 2026; text kept
    ("while capturing \\textbf{⟨⟩\\% of true human classification errors}", CRC + "test_error_recall", pct1v),
    ("Auto-approval rate & $⟨⟩\\%$ & $33.7\\%$", CRC + "test_auto_approval_rate", pct1v),
    ("Error recall & $⟨⟩\\%$ & $64.7\\%$", CRC + "test_error_recall", pct1v),
    ("Efficiency lift vs.\\ random & $⟨⟩\\times$", CRC + "test_efficiency_lift", lambda x: f"{x:.2f}"),
    # --- the error meta-classifier (AWLF)
    ("achieves a mean $\\text{ROC-AUC} = ⟨⟩$ across folds", AW + "Logistic Regression (AWLF)|mean_auc", f4),
    ("the single fit on the full cohort scores $⟨⟩$;", "exp20_error_awlf_results.json:error_awlf_performance|roc_auc", f4),
    ("scored $⟨⟩$ and $0.9908$;", AW + "Gradient Boosting|mean_auc", f4),
    ("scored $0.9905$ and $⟨⟩$;", AW + "Random Forest|mean_auc", f4),

]


def main():
    text = "\n".join(p.read_text() for p in FILES)
    bad = []
    for ctx, spec, fmt in ENTRIES:
        try:
            filled = ctx.replace(M, fmt(val(spec)))
        except Exception as e:
            bad.append(f"cannot read {spec}: {e}"); continue
        n = text.count(filled)
        if n != 1:
            bad.append(f"found {n}x: {filled!r}  <- {spec}")
    for b in bad:
        print("  " + b)
    print(f"{len(ENTRIES) - len(bad)} of {len(ENTRIES)} in-sample (C1) figures match their result files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
