#!/bin/bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate thesis_env

DATA_DIR=~/thesis_project/data/colibri/images/data
OUT_DIR=~/thesis_project/results_full_parallel
LOGFILE=~/thesis_project/logs/full_pipeline_colibri_parallel.log

mkdir -p $(dirname $LOGFILE)
mkdir -p $OUT_DIR
echo "🚀 Starting Parallel Colibri Pipeline at $(date)" | tee -a $LOGFILE

# --- GPU Assignment ------------------------------------------------------------
# Adjust based on available GPUs
GPU0=0
GPU1=1
GPU2=2
GPU3=3

# 1️⃣ CycleGAN Translation (Auto-detect checkpoint)
echo "[1/14] Starting CycleGAN translation (auto-detecting checkpoint)..." | tee -a $LOGFILE

# Detect the newest available hist2modern_* checkpoint folder
CYCLEGAN_CKPT=$(find ~/thesis_project/pytorch-CycleGAN-and-pix2pix/checkpoints -maxdepth 1 -type d -name "hist2modern*" | sort | tail -n 1)

if [ -z "$CYCLEGAN_CKPT" ]; then
  echo "❌ No CycleGAN checkpoint found! Please train or copy one to checkpoints/." | tee -a $LOGFILE
  exit 1
else
  echo "✅ Using CycleGAN checkpoint: $CYCLEGAN_CKPT" | tee -a $LOGFILE
  CYCLEGAN_NAME=$(basename "$CYCLEGAN_CKPT")
fi

python ~/thesis_project/pytorch-CycleGAN-and-pix2pix/test.py \
  --dataroot $DATA_DIR \
  --name $CYCLEGAN_NAME \
  --model test \
  --dataset_mode single \
  --results_dir $OUT_DIR/cyclegan_modern 2>&1 | tee -a $LOGFILE

# --- Step 2: Real-ESRGAN Restoration (GPU1) -----------------------------------
echo "[2/14] Real-ESRGAN Restoration on GPU$GPU1..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU1 python ~/thesis_project/scripts/run_realesrgan_restoration.py \
  --input $DATA_DIR \
  --output $OUT_DIR/restored 2>&1 | tee -a $LOGFILE &

# Wait for both pre-processing jobs to finish
wait
echo "✅ Pre-processing complete at $(date)" | tee -a $LOGFILE

# --- Step 3: BLIP-2 Captioning (GPU0) ----------------------------------------
echo "[3/14] Running BLIP-2 Captioning on GPU$GPU0..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU0 python ~/thesis_project/scripts/run_blip2_captioning.py \
  --input $OUT_DIR/restored \
  --output $OUT_DIR/blip2_captions 2>&1 | tee -a $LOGFILE &

# --- Step 4: GroundingDINO (GPU1) --------------------------------------------
echo "[4/14] GroundingDINO Detection on GPU$GPU1..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU1 python ~/thesis_project/scripts/run_groundingdino_restored.py \
  --input $OUT_DIR/restored \
  --output $OUT_DIR/groundingdino 2>&1 | tee -a $LOGFILE &

# --- Step 5: OWL-ViT (GPU2) ---------------------------------------------------
echo "[5/14] OWL-ViT Detection on GPU$GPU2..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU2 python ~/thesis_project/scripts/run_owlvit_v2.py \
  --input $OUT_DIR/restored \
  --output $OUT_DIR/owlvit 2>&1 | tee -a $LOGFILE &

wait
echo "✅ Captioning + Detection stages finished at $(date)" | tee -a $LOGFILE

# --- Step 6: Pseudo-label Fusion (CPU) ----------------------------------------
echo "[6/14] Fusing pseudo-labels..." | tee -a $LOGFILE
python ~/thesis_project/scripts/fuse_pseudo_labels_restored.py \
  --output $OUT_DIR/pseudo_labels_fused.json 2>&1 | tee -a $LOGFILE

# --- Step 7: YOLOv11 Training (GPU3) ------------------------------------------
echo "[7/14] YOLOv11 Fine-tuning on GPU$GPU3..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU3 python ~/thesis_project/scripts/train_yolo11_final.py \
  --data $OUT_DIR/pseudo_labels_fused.json \
  --output $OUT_DIR/yolo11_runs 2>&1 | tee -a $LOGFILE

# --- Step 8: Scene Classification (CLIP + SigLIP) -----------------------------
echo "[8/14] Scene Classification on GPU$GPU1..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU1 python ~/thesis_project/scripts/scene_classification_clip_siglip.py \
  --input $OUT_DIR/restored \
  --output $OUT_DIR/scene_labels 2>&1 | tee -a $LOGFILE

# --- Step 9: Agreement Analysis -----------------------------------------------
echo "[9/14] Computing Agreement Metrics..." | tee -a $LOGFILE
python ~/thesis_project/scripts/align_detections_for_agreement.py \
  --input $OUT_DIR \
  --output $OUT_DIR/agreement 2>&1 | tee -a $LOGFILE

# --- Step 10: Visualization ---------------------------------------------------
echo "[10/14] Generating Visualizations..." | tee -a $LOGFILE
python ~/thesis_project/scripts/generate_agreement_collages.py \
  --input $OUT_DIR/agreement \
  --output $OUT_DIR/visuals 2>&1 | tee -a $LOGFILE

# --- Step 11: LLaVA-OneVision Reasoning (GPU2) -------------------------------
echo "[11/14] Running LLaVA VQA on GPU$GPU2..." | tee -a $LOGFILE
CUDA_VISIBLE_DEVICES=$GPU2 python ~/thesis_project/scripts/run_llava_vqa.py \
  --input $OUT_DIR/restored \
  --output $OUT_DIR/vqa 2>&1 | tee -a $LOGFILE

# --- Step 12: Gradio Interface (Optional) ------------------------------------
echo "[12/14] Launching Gradio Interface..." | tee -a $LOGFILE
PORT=$(shuf -i 7860-7890 -n 1)
export GRADIO_SERVER_PORT=$PORT
CUDA_VISIBLE_DEVICES=$GPU0 python ~/thesis_project/scripts/gradio_interface.py \
  --port $PORT 2>&1 | tee -a $LOGFILE &

# --- Step 13: Zenodo Export ---------------------------------------------------
echo "[13/14] Exporting Dataset for Zenodo..." | tee -a $LOGFILE
python ~/thesis_project/scripts/export_to_zenodo.py \
  --input $OUT_DIR \
  --output $OUT_DIR/zenodo 2>&1 | tee -a $LOGFILE

echo "✅ All stages completed successfully at $(date)" | tee -a $LOGFILE