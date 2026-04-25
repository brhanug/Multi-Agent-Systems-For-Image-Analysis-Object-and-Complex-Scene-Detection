#!/bin/bash
# Final Pipeline Orchestrator: Open-Vocabulary Phase

set -e

BASE_DIR="/data/brhanu/thesis_project"
CONDA_PY="/home/brhanu/miniconda/envs/thesis_env/bin/python"

echo "=========================================================="
echo "🚀 PHASE 2: OPEN-VOCABULARY PIPELINE STARTED"
echo "=========================================================="

echo "⏳ Step 1: Waiting for GroundingDINO Extraction to Complete..."
# We don't need to run it again, we just wait for the current nohup process to finish.
# We will find the PID of the generate_open_vocab_labels.py script and wait for it.
PID=$(pgrep -f "generate_open_vocab_labels.py" || echo "")

if [ -n "$PID" ]; then
    echo "🔍 Found active GroundingDINO process (PID: $PID). Waiting for it to finish..."
    tail --pid=$PID -f /dev/null
    echo "✅ Label Extraction Finished."
else
    echo "✅ No active GroundingDINO process found. Assuming extraction is complete."
fi

echo "=========================================================="
echo "🚀 Step 2: Initiating YOLOv11 Open-Vocabulary Training"
echo "=========================================================="
$CONDA_PY $BASE_DIR/scripts/core_pipeline/train_open_vocab.py

echo "=========================================================="
echo "🎉 FINAL PIPELINE COMPLETE!"
echo "=========================================================="
