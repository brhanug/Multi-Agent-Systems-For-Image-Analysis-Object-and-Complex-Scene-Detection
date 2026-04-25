#!/bin/bash

# Configuration
PROJECT_ROOT="/data/brhanu/thesis_project"
CONDA_ENV="/data/brhanu/miniconda/envs/thesis_env"
PYTHON_BIN="$CONDA_ENV/bin/python3"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo "🚀 Starting Visual Historian Services..."

# 1. Start vLLM Server (on GPU 3)
echo "🔹 Starting vLLM Server on GPU 3 (Port 8000)..."
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --port 8000 \
    --limit-mm-per-prompt '{"image": 1}' \
    > "$LOG_DIR/vllm_server_direct.log" 2>&1 &

# 2. Wait for vLLM to initialize (can take 1-2 minutes)
echo "⏳ Waiting for vLLM Server to be ready..."
RETRIES=0
MAX_RETRIES=30
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 10
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -gt $MAX_RETRIES ]; then
        echo "⚠️ vLLM Server taking too long or failed to start. Check logs/vllm_server_direct.log"
        # We continue anyway to start other services
        break
    fi
    echo "   ...still waiting ($((RETRIES * 10))s)..."
done
echo "✅ vLLM Server is UP!"

# 3. Start VQA Interface (Port 7860)
echo "🔹 Starting VQA Interface (Port 7860)..."
nohup $PYTHON_BIN "$PROJECT_ROOT/scripts/analysis_viz/vqa_interface.py" \
    > "$LOG_DIR/vqa_interface.log" 2>&1 &

# 4. Start Archive Explorer (Port 7862)
echo "🔹 Starting Archive Explorer (Port 7862)..."
nohup $PYTHON_BIN "$PROJECT_ROOT/scripts/analysis_viz/archive_visualizer.py" \
    > "$LOG_DIR/archive_visualizer.log" 2>&1 &

# 5. Start Digital Hermeneutics Sandbox (Port 7864)
echo "🔹 Starting Digital Hermeneutics Sandbox (Port 7864)..."
nohup $PYTHON_BIN "$PROJECT_ROOT/scripts/analysis_viz/run_rag_hermeneutics.py" \
    > "$LOG_DIR/rag_hermeneutics.log" 2>&1 &

# 6. Start Ngrok Tunnels
echo "🔹 Starting Ngrok Tunnels..."
nohup $PYTHON_BIN "$PROJECT_ROOT/start_tunnels.py" \
    > "$LOG_DIR/ngrok_startup_final.log" 2>&1 &

echo "✨ All services have been initiated."
echo "📜 Check 'logs/' directory for output from each service."
