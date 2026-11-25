#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full multi-agent pipeline entry point
#
# Usage:
#   ./run_pipeline.sh [--config config.yaml] [--base-dir /path/to/data]
#
# Steps:
#   1. OWL-ViT batch inference   (src/owlvit_batch.py)
#   2. Pseudo-label generation   (src/pseudo_label_generator.py)
#   3. Multi-agent validation    (scripts/core_pipeline/run_multi_agent_validation.py)
#   4. Agent comparison          (scripts/core_pipeline/run_agent_comparison.py)
#   5. Error analysis            (scripts/core_pipeline/run_error_analysis.py)
#
# Each step is skipped automatically if its output already exists (idempotent).
# =============================================================================

set -euo pipefail

CONFIG="config.yaml"
BASE_DIR="/data/brhanu/thesis_project"
DATASET_ROOT="final_dataset"
OUTPUT_DIR="results/multi_agent"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";     shift 2 ;;
        --base-dir)  BASE_DIR="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "================================================================"; echo "  STEP $*"; echo "================================================================"; }

RESULTS="$BASE_DIR/$OUTPUT_DIR"
mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Step 1: OWL-ViT inference
# ---------------------------------------------------------------------------
OWLVIT_CSV="$BASE_DIR/results/owlvit_results.csv"
step "1 — OWL-ViT batch inference"
if [[ -f "$OWLVIT_CSV" ]]; then
    log "Output already exists at $OWLVIT_CSV — skipping (delete to re-run)."
else
    OUTPUT_CSV="$OWLVIT_CSV" python src/owlvit_batch.py
    log "Done."
fi

# ---------------------------------------------------------------------------
# Step 2: Pseudo-label generation
# ---------------------------------------------------------------------------
step "2 — Pseudo-label generation"
PSEUDO_DIR="$BASE_DIR/data/pseudo_labels"
if [[ -d "$PSEUDO_DIR" ]] && [[ "$(ls -A "$PSEUDO_DIR" 2>/dev/null)" ]]; then
    log "Pseudo-labels directory non-empty ($PSEUDO_DIR) — skipping."
else
    OWLVIT_CSV="$OWLVIT_CSV" PSEUDO_LABEL_DIR="$PSEUDO_DIR" python src/pseudo_label_generator.py
    log "Done."
fi

# ---------------------------------------------------------------------------
# Step 3: Multi-agent validation
# ---------------------------------------------------------------------------
step "3 — Multi-agent validation"
MAV_CSV="$RESULTS/multi_agent_validation_scores.csv"
if [[ -f "$MAV_CSV" ]]; then
    log "Output already exists at $MAV_CSV — skipping."
else
    python scripts/core_pipeline/run_multi_agent_validation.py \
        --config "$CONFIG" \
        --base-dir "$BASE_DIR" \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$OUTPUT_DIR"
    log "Done."
fi

# ---------------------------------------------------------------------------
# Step 4: Agent comparison
# ---------------------------------------------------------------------------
step "4 — Agent comparison"
CMP_CSV="$RESULTS/agent_comparison_scores.csv"
if [[ -f "$CMP_CSV" ]]; then
    log "Output already exists at $CMP_CSV — skipping."
else
    python scripts/core_pipeline/run_agent_comparison.py \
        --config "$CONFIG" \
        --base-dir "$BASE_DIR" \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$OUTPUT_DIR"
    log "Done."
fi

# ---------------------------------------------------------------------------
# Step 5: Error analysis
# ---------------------------------------------------------------------------
step "5 — Error / failure-mode analysis"
ERR_CSV="$RESULTS/error_analysis.csv"
if [[ -f "$ERR_CSV" ]]; then
    log "Output already exists at $ERR_CSV — skipping."
else
    python scripts/core_pipeline/run_error_analysis.py \
        --base-dir "$BASE_DIR" \
        --scores-csv "$OUTPUT_DIR/agent_comparison_scores.csv" \
        --output-dir "$OUTPUT_DIR"
    log "Done."
fi

echo
log "All pipeline steps complete.  Results in: $RESULTS"
