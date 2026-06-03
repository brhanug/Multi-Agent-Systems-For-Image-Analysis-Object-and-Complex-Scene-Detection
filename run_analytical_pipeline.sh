#!/bin/bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate thesis_env

LOGFILE=~/thesis_project/logs/analytical_pipeline.log
mkdir -p $(dirname $LOGFILE)

echo "🚀 Starting 6-Agent Analytical Pipeline at $(date)" | tee $LOGFILE

# 1. Temporal Historian
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[1/7] Agent 1: Temporal Historian" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_ppn_temporal_analysis.py 2>&1 | tee -a $LOGFILE

# 2. Retrieval Agent
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[2/7] Agent 2: Retrieval Agent" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_cross_lingual_retrieval.py 2>&1 | tee -a $LOGFILE

# 3. Demographic Profiler
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[3/7] Agent 4: Demographic Profiler" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_demographic_profiler.py 2>&1 | tee -a $LOGFILE

# 4. Geospatial Analyst
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[4/7] Agent 5: Geospatial Analyst" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_geospatial_analyst.py 2>&1 | tee -a $LOGFILE

# 5. Adjudicator (Macro Fusion)
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[5/7] Central Coordinator: Active Agent Adjudicator" | tee -a $LOGFILE
python3 scripts/core_pipeline/active_agent_adjudicator.py 2>&1 | tee -a $LOGFILE

# 6. Ablation Study
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[6/7] Scientific Evaluation: 6-Agent Macro Ablation" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_6_agent_ablation.py 2>&1 | tee -a $LOGFILE

# 7. Uncertainty Calibration
echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "[7/7] Scientific Evaluation: Uncertainty Calibration" | tee -a $LOGFILE
python3 scripts/core_pipeline/run_calibration_metrics.py 2>&1 | tee -a $LOGFILE

echo "--------------------------------------------------------" | tee -a $LOGFILE
echo "✅ Analytical Pipeline Completed at $(date)" | tee -a $LOGFILE
