#!/usr/bin/env bash
# Full Tier 1 re-run over all 12,110 corpus images, every model, from scratch.
# Detached and resumable: each stage skips ids already written, so re-running
# this script after an interruption continues rather than restarting.
set -u
BASE=/data/brhanu/thesis_project
PY=$BASE/.venv/bin/python3
LOG=$BASE/results/signal_rebuild/full_tier1_run.log
mkdir -p "$(dirname "$LOG")"
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

say "=== FULL TIER 1 RUN START (4x A6000) ==="

# ---- Stage 1: LLaVA-OneVision VQA, 4 shards, one per GPU ----
say "STAGE 1  LLaVA-OneVision  (4 shards)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY $BASE/scripts/signal_rebuild/run_llava_full_corpus.py \
    --shard $i --num-shards 4 >> "$BASE/results/signal_rebuild/llava_shard_$i.log" 2>&1 &
done
wait
say "STAGE 1  done -> $(cat $BASE/results/signal_rebuild/llava_full/shard_*.jsonl 2>/dev/null | wc -l) rows"

# ---- Stage 2: Kosmos-2.5 grounded OCR, 4 shards ----
say "STAGE 2  Kosmos-2.5  (4 shards)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY $BASE/scripts/signal_rebuild/run_kosmos_full_corpus.py \
    --shard $i --num-shards 4 >> "$BASE/results/signal_rebuild/kosmos_shard_$i.log" 2>&1 &
done
wait
say "STAGE 2  done -> $(cat $BASE/results/signal_rebuild/kosmos_full/shard_*.jsonl 2>/dev/null | wc -l) rows"

# ---- Stage 3: SigLIP scene classifier, corpus-wide ----
say "STAGE 3  SigLIP scene"
SIG=$BASE/results/signal_rebuild/siglip_scene_full.jsonl
[ -f "$SIG" ] && mv "$SIG" "${SIG%.jsonl}_previous.jsonl" && say "  archived previous SigLIP output"
CUDA_VISIBLE_DEVICES=0 $PY $BASE/scripts/signal_rebuild/run_siglip_scene_full_corpus.py \
  >> "$BASE/results/signal_rebuild/siglip_run.log" 2>&1
say "STAGE 3  done -> $(wc -l < $SIG 2>/dev/null || echo 0) rows"

# ---- Stage 4: YOLO detector (already complete; resumes only if short) ----
say "STAGE 4  YOLOv11 detector"
$PY $BASE/scripts/signal_rebuild/run_yolo_full_corpus.py --device 0 --batch 48 \
  >> "$BASE/results/signal_rebuild/yolo_run.log" 2>&1
say "STAGE 4  done -> $(wc -l < $BASE/results/signal_rebuild/yolo_full_corpus.jsonl 2>/dev/null || echo 0) rows"

say "=== ALL STAGES COMPLETE ==="
