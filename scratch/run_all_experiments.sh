#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Run all thesis evaluation pipelines from scratch
# =============================================================================
set -e
set -x

echo "🚀 Starting full Multi-Agent and Curation pipeline..."

# Ensure we are using the virtual environment
source .venv/bin/activate

# 1. Multi-agent validation
python3 scripts/core_pipeline/run_multi_agent_validation.py \
    --config config.yaml \
    --base-dir /data/brhanu/thesis_project \
    --dataset-root final_dataset \
    --output-dir results/multi_agent

# 2. Agent comparison
python3 scripts/core_pipeline/run_agent_comparison.py \
    --config config.yaml \
    --base-dir /data/brhanu/thesis_project \
    --dataset-root final_dataset \
    --output-dir results/multi_agent

# 3. Error / failure-mode analysis
python3 scripts/core_pipeline/run_error_analysis.py \
    --base-dir /data/brhanu/thesis_project \
    --scores-csv results/multi_agent/agent_comparison_scores.csv \
    --output-dir results/multi_agent

# 4. Temporal claim analysis
python3 scripts/core_pipeline/run_ppn_temporal_analysis.py

# 5. Retrieval (run_cross_lingual_retrieval.py removed -- fabricated, simulated
#    embeddings, MAP=0.820/nDCG=0.870 withdrawn; see run_analytical_pipeline.sh)
python3 scripts/core_pipeline/run_semantic_retrieval_eval.py

# 6. Demographic profiling
python3 scripts/core_pipeline/run_demographic_profiler.py

# 7. Geospatial analysis
python3 scripts/core_pipeline/run_geospatial_analyst.py

# 8. Active agent adjudicator
python3 scripts/core_pipeline/active_agent_adjudicator.py

# 9. 6-agent ablation
python3 scripts/core_pipeline/run_6_agent_ablation.py

# 10. Calibration metrics
python3 scripts/core_pipeline/run_calibration_metrics.py

# 11. Agent diversity analysis
python3 scripts/core_pipeline/run_agent_diversity_analysis.py

# 12. Agent independence analysis
python3 scripts/core_pipeline/run_agent_independence_analysis.py

# 13. Box-level IoU vs SAA uncertainty
python3 scripts/core_pipeline/run_box_level_iou_uncertainty.py

# 14. Complexity analysis
python3 scripts/core_pipeline/run_complexity_analysis.py

# 15. Scene type analysis
python3 scripts/core_pipeline/run_scene_type_analysis.py

# 16. Final viva experiments compilation
python3 scripts/core_pipeline/run_detailed_viva_experiments.py

# 17. Regenerate thesis figures and copy to latex_assets
python3 scripts/analysis_viz/generate_thesis_figures_extended.py
cp results/figures/thesis_agent_diversity_heatmap.png latex_assets/thesis_agent_diversity_heatmap.png

echo "✅ All experiments executed successfully and all output matrices and assets refreshed."

