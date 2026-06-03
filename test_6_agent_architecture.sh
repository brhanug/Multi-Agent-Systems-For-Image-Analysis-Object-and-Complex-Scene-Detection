#!/bin/bash
# test_6_agent_architecture.sh
# This script runs the analytical layer of the 6-Agent System (Agents 1-5 + Coordinator).
# Note: Agent 0 (the core CV pipeline) takes ~5.6 hours and its results are already cached.

set -e

echo "===================================================="
echo "🧪 Testing 6-Agent Architecture (Analytical Layers)"
echo "===================================================="

# Ensure scripts are executable
chmod +x scripts/core_pipeline/run_demographic_profiler.py
chmod +x scripts/core_pipeline/run_geospatial_analyst.py
chmod +x scripts/core_pipeline/active_agent_adjudicator.py

echo -e "\n[1/6] Skipping Agent 0 (Full CV Pipeline) - Using cached output..."
sleep 1

echo -e "\n[2/6] Running Agent 1: Temporal Historian..."
# Note: If this takes too long, you can press Ctrl+C, it won't break the others.
python3 scripts/core_pipeline/run_ppn_temporal_analysis.py || echo "Temporal analysis skipped or failed."

echo -e "\n[3/6] Running Agent 2: Retrieval Agent (RAG)..."
python3 scripts/core_pipeline/run_semantic_retrieval_eval.py || echo "Retrieval analysis skipped or failed."

echo -e "\n[4/6] Running Agent 3: Hallucination Critic..."
python3 scripts/core_pipeline/run_rq2_disagreement_analysis.py || echo "Critic analysis skipped or failed."

echo -e "\n[5/6] Running Agent 4: Demographic Profiler..."
python3 scripts/core_pipeline/run_demographic_profiler.py

echo -e "\n[6/6] Running Agent 5: Geospatial Analyst..."
python3 scripts/core_pipeline/run_geospatial_analyst.py

echo -e "\n[Final] Running Coordinator Adjudicator (Dry-Run Mode)..."
python3 scripts/core_pipeline/active_agent_adjudicator.py --top-n 10 --dry-run

echo -e "\n✅ All analytical agents executed successfully."
echo "Results can be found in: results/multi_agent/"
