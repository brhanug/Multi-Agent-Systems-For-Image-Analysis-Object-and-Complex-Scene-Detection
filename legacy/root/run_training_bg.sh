#!/bin/bash
nohup /data/brhanu/miniconda/envs/thesis_env/bin/python scripts/core_pipeline/train_yolo11_final.py > logs/yolo11_final_refresh.log 2>&1 &
echo "Training started with thesis_env python. Check logs/yolo11_final_refresh.log for progress."
echo "PID: $!"
