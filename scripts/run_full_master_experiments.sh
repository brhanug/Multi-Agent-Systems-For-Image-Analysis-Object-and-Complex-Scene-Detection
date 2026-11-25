#!/bin/bash
# =============================================================================
# run_full_master_experiments.sh — Run all validation and curation pipelines
# =============================================================================
set -e
set -x

echo "🚀 Starting Master Multi-Agent Pipeline Refresh..."

# Use the local virtualenv python interpreter directly to avoid broken activate paths
PYTHON_BIN=/data/brhanu/thesis_project/.venv/bin/python3

# 1. Agent 0 Foundation & Validation Core
$PYTHON_BIN scripts/core_pipeline/run_agent_comparison.py
$PYTHON_BIN scripts/core_pipeline/run_multi_agent_validation.py
$PYTHON_BIN scripts/core_pipeline/fix_correlated_errors.py

# 1b. Clean group-level (publication-disjoint) held-out evaluation.
# Rebuilds the frozen train/calibration/test split fresh each run so it can
# never silently go stale like the old SCI-tercile routing figure did.
# run_scene_complexity_index.py (the actual SCI generator) was previously
# never wired into this pipeline at all -- only its downstream consumer
# (run_complexity_analysis.py, below) was -- which is exactly how the SCI
# formula drifted out from under the routing policy unnoticed. It must run
# before run_clean_holdout_evaluation.py, which depends on its output.
# run_clean_holdout_evaluation.py also now emits negative_control_decomposition
# (SAA disagreement AUROC split by true class: pos 0.980 / neg 0.020) -- the
# thesis's central "Negative-Control Generalization Gap" table. This exact
# computation was previously done once, ad hoc, with no backing script, so it
# could not be regenerated or re-verified by anyone re-running this pipeline.
# Restored into the script itself during the full-project re-run audit; the
# original hand-copied numbers were independently re-derived and matched
# exactly (0.9796->0.980, 0.0201->0.020), so no thesis correction was needed.
$PYTHON_BIN scripts/core_pipeline/run_scene_complexity_index.py
$PYTHON_BIN scripts/core_pipeline/build_clean_holdout_split.py
$PYTHON_BIN scripts/core_pipeline/run_clean_holdout_evaluation.py
# exp29_conformal_risk_control.py was, like run_scene_complexity_index.py
# above, never actually wired into this pipeline despite being the source of
# a headline thesis claim (CRC 87.5%/94.3%/7.57x) -- it calibrates/tests on a
# random 50/50 split of the same positive-heavy C1 cohort, which is exactly
# the leakage class the group-level split above exists to rule out. Both the
# original (in-sample) and clean (group-level) versions are now run and
# reported side by side so neither number can silently go stale again.
$PYTHON_BIN scripts/core_pipeline/exp29_conformal_risk_control.py
$PYTHON_BIN scripts/core_pipeline/run_clean_holdout_crc.py

# 1c. Illustrative frontier-routing extrapolation (Exp 26), calibrated to the
# real, live-API-measured local-vs-frontier F1 gap in coordinator_comparison.json.
# NOTE: coordinator_comparison.json itself is NOT regenerated here -- it comes
# from compare_coordinators.py, which makes live paid Gemini/Claude API calls
# and is intentionally left orphaned from this pipeline so a routine re-run
# never silently re-spends money. If coordinator_comparison.json is missing,
# this step prints a warning and skips rather than fabricating a substitute.
$PYTHON_BIN scripts/core_pipeline/exp26_hybrid_frontier_routing.py

# 2. Agent 1: Temporal Historian
$PYTHON_BIN scripts/core_pipeline/run_ppn_temporal_analysis.py

# 3. Agent 2: Retrieval Agent
$PYTHON_BIN scripts/core_pipeline/run_semantic_retrieval_eval.py
# run_cross_lingual_retrieval.py removed from the pipeline during a full-project
# audit: it never loads or queries any real cross-lingual embedding model (its own
# docstring: "we are avoiding downloading large transformer models... this script
# simulates the embedding alignment"). Its "retrieval" filters a table by labels
# that are true by construction, over an already-synthetic dataset
# (synthetic_human_baseline.csv), and its "realistic" 0.8/0.9 precision values are
# produced by hash(query) % 2, not any measured error. Not cited anywhere in the
# thesis, slides, or any other document, and nothing downstream consumes its
# output -- a real cross-lingual retrieval pilot remains future work.

# 4. Agent 3: Bayesian Pre-screening Ranker & Agreement Audit
$PYTHON_BIN scripts/core_pipeline/run_rq2_disagreement_analysis.py
$PYTHON_BIN scripts/core_pipeline/run_spatial_agreement_audit.py

# 5. Agent 4: Demographic Profiler
$PYTHON_BIN scripts/core_pipeline/run_demographic_profiler.py

# 6. Agent 5: Geospatial Analyst
$PYTHON_BIN scripts/core_pipeline/run_geospatial_analyst.py

# 7. Agent 6: Active Agent Adjudicator
$PYTHON_BIN scripts/core_pipeline/active_agent_adjudicator.py

# 8. Tier 1 Ablation Study
# run_6_agent_ablation.py (deleted) fabricated 3 of its 6 "agents" via a
# hash-based pseudo-random formula and labeled rows with Tier-2 role names
# that did not match the (also unrelated) data actually loaded for them --
# despite being wired into this pipeline and regenerating on every run.
# Replaced with a leave-one-out ablation over the four real Tier 1 agents,
# the only agents that participate in the core SAA/uncertainty calculation
# per the Architectural Decoupling Rule.
$PYTHON_BIN scripts/core_pipeline/run_tier1_ablation.py

# 9. Evaluation and Research Questions (RQ1 - RQ8)
$PYTHON_BIN scripts/core_pipeline/run_research_evaluation.py
$PYTHON_BIN scripts/core_pipeline/run_statistical_report.py
$PYTHON_BIN scripts/core_pipeline/run_complexity_analysis.py
$PYTHON_BIN scripts/core_pipeline/run_gold_simulation.py
$PYTHON_BIN scripts/core_pipeline/run_hitl_simulation.py
$PYTHON_BIN scripts/core_pipeline/run_taxonomy_analysis.py
$PYTHON_BIN scripts/core_pipeline/run_domain_adaptation_analysis.py

# 10. Human Baselines & AWLF Calibration
# NOTE: run_synthetic_worksheet_fill.py and run_awlf_pilot.py were removed —
# they depended on human_baseline_gold_kit/annotator_A/B_worksheet.csv, which
# was synthetically fabricated data presented as real dual-annotator labels
# (see prepare_annotator_worksheets.py, now also removed). AWLF is evaluated
# for real via exp20_error_targeted_awlf.py (AWLF v2) instead.
$PYTHON_BIN scripts/core_pipeline/run_synthetic_human_baseline.py
$PYTHON_BIN scripts/core_pipeline/run_gold_evaluation.py

# 11. Custom Real Expert Spatial Audits (300-image Label Studio cohort)
$PYTHON_BIN scripts/core_pipeline/run_spatial_error_prediction_100.py
$PYTHON_BIN scripts/core_pipeline/run_box_level_iou_uncertainty.py
$PYTHON_BIN scripts/core_pipeline/run_iou_error_taxonomy.py

# 12. Calibration, Diversity, Independence & Scene Type Analysis
$PYTHON_BIN scripts/core_pipeline/run_calibration_metrics.py
$PYTHON_BIN scripts/core_pipeline/run_agent_diversity_analysis.py
$PYTHON_BIN scripts/core_pipeline/run_agent_independence_analysis.py
$PYTHON_BIN scripts/core_pipeline/run_pairwise_agreement_heatmap.py
$PYTHON_BIN scripts/core_pipeline/run_scene_type_analysis.py

# 13. Compilation of Final Viva Report
$PYTHON_BIN scripts/core_pipeline/run_detailed_viva_experiments.py
$PYTHON_BIN scripts/core_pipeline/run_metric_fact_check.py

# 14. CORRECTED-LABEL CHAIN -- produces every headline number in the current thesis
#     (SAA 0.860 overall, 0.9687/0.1152 positive/negative, AWLF 0.8702, and the
#     CRC result: R=0.0201, 60.6% approval @ 91.1% recall, 2.31-3.09x lift).
#     Added after a provenance audit found this chain was never wired in, so a
#     full pipeline run regenerated only the superseded pre-correction results.
$PYTHON_BIN scripts/core_pipeline/build_corrected_negative_control_pool.py --verify
$PYTHON_BIN scripts/core_pipeline/build_corrected_clean_holdout_split.py
$PYTHON_BIN scripts/core_pipeline/run_corrected_clean_holdout_evaluation.py
$PYTHON_BIN scripts/core_pipeline/run_corrected_clean_holdout_crc.py

# 15. Gold-labeled Tier 2 agent precision + Europeana transfer ground truth
$PYTHON_BIN scripts/core_pipeline/run_tier2_agent_gold_precision.py --verify
$PYTHON_BIN scripts/core_pipeline/run_europeana_transfer_ground_truth_eval.py

# 16. Ensemble-heterogeneity precondition (the diagnostic derived from the
#     Europeana failure): Colibri is the working control, Europeana the
#     collapsed case. Requires no labels.
$PYTHON_BIN scripts/core_pipeline/run_ensemble_heterogeneity_diagnostic.py \
  --scores-csv results/multi_agent/agent_comparison_scores.csv \
  --label "Colibri C0 (deployment corpus)" \
  --out-json results/multi_agent/ensemble_heterogeneity_colibri.json
$PYTHON_BIN scripts/core_pipeline/run_ensemble_heterogeneity_diagnostic.py \
  --scores-csv results/external_archive/europeana_real_pilot/europeana_real_agent_scores_FINAL.csv \
  --label "Europeana C5 (transfer pilot)" \
  --out-json results/external_archive/europeana_real_pilot/ensemble_heterogeneity_europeana.json

echo "✅ All experiments executed successfully and all output matrices refreshed."

# ---------------------------------------------------------------------------
# 17. What U_SAA actually measures: derived-term dependency + baseline comparison
#     (Section 5.2 / Table 5.1 / Slide A.15)
# ---------------------------------------------------------------------------
echo "[17/17] Uncertainty signal decomposition"
$PYTHON_BIN scripts/core_pipeline/run_uncertainty_signal_decomposition.py
