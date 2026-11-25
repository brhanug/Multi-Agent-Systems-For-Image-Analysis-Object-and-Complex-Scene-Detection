#!/bin/bash
nohup python scripts/core_pipeline/finalize_v1_package.py \
    --flatten \
    --weights results/yolo11_final_v1_refresh/exp_final2/weights/best.pt \
    > logs/finalize_v1_package.log 2>&1 &
echo "Finalization started. PID: $!"
