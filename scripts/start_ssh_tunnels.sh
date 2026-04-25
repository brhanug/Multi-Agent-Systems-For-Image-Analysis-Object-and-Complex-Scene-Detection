#!/bin/bash

# Kill any existing ssh tunnels to these ports
pkill -f "localhost.run" || true

echo "Starting SSH Tunnels via localhost.run..."

# VQA (7860)
nohup ssh -o StrictHostKeyChecking=no -R 80:localhost:7860 nokey@localhost.run > /data/brhanu/thesis_project/logs/tunnel_vqa.log 2>&1 &
# Explorer (7862)
nohup ssh -o StrictHostKeyChecking=no -R 80:localhost:7862 nokey@localhost.run > /data/brhanu/thesis_project/logs/tunnel_explorer.log 2>&1 &
# RAG (7864)
nohup ssh -o StrictHostKeyChecking=no -R 80:localhost:7864 nokey@localhost.run > /data/brhanu/thesis_project/logs/tunnel_rag.log 2>&1 &
# vLLM (8000)
nohup ssh -o StrictHostKeyChecking=no -R 80:localhost:8000 nokey@localhost.run > /data/brhanu/thesis_project/logs/tunnel_vllm.log 2>&1 &

echo "Tunnels initiated. Waiting for URLs..."
sleep 10

echo "--- VQA URL ---"
grep -o 'https://[^ ]*\.lhr\.life' /data/brhanu/thesis_project/logs/tunnel_vqa.log || grep -o 'https://[^ ]*\.localhost\.run' /data/brhanu/thesis_project/logs/tunnel_vqa.log
echo "--- Explorer URL ---"
grep -o 'https://[^ ]*\.lhr\.life' /data/brhanu/thesis_project/logs/tunnel_explorer.log || grep -o 'https://[^ ]*\.localhost\.run' /data/brhanu/thesis_project/logs/tunnel_explorer.log
echo "--- RAG URL ---"
grep -o 'https://[^ ]*\.lhr\.life' /data/brhanu/thesis_project/logs/tunnel_rag.log || grep -o 'https://[^ ]*\.localhost\.run' /data/brhanu/thesis_project/logs/tunnel_rag.log
echo "--- vLLM URL ---"
grep -o 'https://[^ ]*\.lhr\.life' /data/brhanu/thesis_project/logs/tunnel_vllm.log || grep -o 'https://[^ ]*\.localhost\.run' /data/brhanu/thesis_project/logs/tunnel_vllm.log
