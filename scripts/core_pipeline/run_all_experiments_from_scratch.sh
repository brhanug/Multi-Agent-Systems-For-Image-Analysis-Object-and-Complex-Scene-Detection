#!/bin/bash
# run_all_experiments_from_scratch.sh
# -----------------------------------
# Orchestrates the full execution of Agents 0-5 and all valuation metrics.
# Incorporates fixes for ID collisions and AWLF refinements.

set -e

BASE_DIR="/data/brhanu/thesis_project"
cd "$BASE_DIR"

echo "🚀 Starting Full Multi-Agent Experiment Refresh (Agents 0-5)..."

# --- AGENT 0: Foundation ---
echo "🤖 Running Agent 0 (Foundation: Restoration, Detection, Scene, VQA)..."
python3 scripts/core_pipeline/run_agent_comparison.py
python3 scripts/core_pipeline/run_multi_agent_validation.py
python3 scripts/core_pipeline/fix_correlated_errors.py

# --- AGENT 1: Temporal Historian ---
echo "📅 Running Agent 1 (Temporal Historian)..."
python3 scripts/core_pipeline/run_ppn_temporal_analysis.py

# --- AGENT 2: Retrieval Agent ---
echo "🔍 Running Agent 2 (Retrieval Agent)..."
python3 scripts/core_pipeline/run_semantic_retrieval_eval.py

# --- AGENT 3: Hallucination Critic (Agreement) ---
echo "⚖️ Running Agent 3 (Hallucination Critic)..."
python3 scripts/core_pipeline/run_rq2_disagreement_analysis.py

# --- AGENT 4: Demographic Profiler ---
echo "👥 Running Agent 4 (Demographic Profiler)..."
python3 scripts/core_pipeline/run_demographic_profiler.py

# --- AGENT 5: Geospatial Analyst ---
echo "🌍 Running Agent 5 (Geospatial Analyst)..."
python3 scripts/core_pipeline/run_geospatial_analyst.py

# --- MACRO ABLATION: 6-Agent Comparison ---
echo "🔬 Running 6-Agent Macro Ablation Study..."
python3 scripts/core_pipeline/run_6_agent_ablation.py

# --- EVALUATION & RESEARCH QUESTIONS ---
echo "📊 Running Evaluation Suite (RQ1-RQ8)..."
python3 scripts/core_pipeline/run_research_evaluation.py
python3 scripts/core_pipeline/run_statistical_report.py
python3 scripts/core_pipeline/run_complexity_analysis.py
python3 scripts/core_pipeline/run_gold_simulation.py
python3 scripts/core_pipeline/run_hitl_simulation.py
python3 scripts/core_pipeline/run_taxonomy_analysis.py
python3 scripts/core_pipeline/run_domain_adaptation_analysis.py

# --- HUMAN BASELINE & AWLF REFINEMENT ---
echo "👤 Running Synthetic Human Baseline & Gold Worksheet Fill..."
python3 scripts/core_pipeline/run_synthetic_human_baseline.py
python3 scripts/core_pipeline/run_synthetic_worksheet_fill.py
python3 scripts/core_pipeline/run_gold_evaluation.py

echo "🎓 Running AWLF (Agreement-Weighted Label Filter) Pilot..."
python3 scripts/core_pipeline/run_awlf_pilot.py

# --- FINAL FACT CHECK ---
echo "✅ Running Final Metric Fact Check..."
python3 scripts/core_pipeline/run_metric_fact_check.py

echo "🏁 All experiments completed from scratch."
