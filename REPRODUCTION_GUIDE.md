# Reproduction Guide: Visual Historian Pipeline

This guide provides instructions for researchers to reproduce the "Pure Pipeline" results and the "Visual Historian" dataset enhancements.

## 1. Environment Setup

The pipeline requires a high-VRAM GPU (A6000 or similar recommended for VLM ensembling).

```bash
# Create and activate the environment
conda create -n thesis_env python=3.10
conda activate thesis_env

# Install core dependencies
pip install ultralytics torch torchvision torchaudio transformers
pip install diffusers accelerate safetensors
pip install gradio pandas numpy matplotlib seaborn tqdm
```

## 2. Dataset Structure

The Zenodo package (`final_dataset.zip`) should be extracted into the project root.

```
/final_dataset/
  /images/
    /original/           - Raw archival scans
    /restored/           - CycleGAN + Real-ESRGAN outputs
    /diffusion_restored/ - Stable Diffusion refined outputs
  /labels/
    /yolo_labels/        - Consolidated YOLOv11 refined labels
  /metadata/
    /manifest.csv        - Master semantic index
```

## 3. Running the Pipeline

### Object Detection (YOLOv11 Inference)
To run the distilled student model on new images:
```bash
python scripts/core_pipeline/run_student_inference.py --source path/to/images --weights weights/student_v3_best.pt
```

### Multimodal Refinement (VLM Ensemble)
To generate pseudo-labels using GroundingDINO and Florence-2:
```bash
python scripts/core_pipeline/run_ensemble_grounding.py --config config/ensemble_v8.yaml
```

### Agreement Analysis
To compute the "Global Mean Agreement" across model outputs:
```bash
python scripts/analysis_viz/analyze_model_agreement_v8.py
```

## 4. Scaling Strategy: Hierarchy of Fidelity

To optimize compute and storage (Zenodo's 50GB limit), the dataset follows a tiered approach:

*   **Tier 1: Full Dataset (12,110 images)**: All images include Original and Restored (CycleGAN+ESRGAN) versions, along with semantic metadata (VQA, Scene Claschecsification, and YOLO predictions).
*   **Tier 2: Refined Subset (6,480 images)**: A subset of images processed through the **VLM Ensemble (Florence-2 and GroundingDINO)** to generate high-confidence pseudo-labels for training.
*   **Tier 3: Human Baseline Audit (2,000 images)**: The ground-truth calibration subset. Selected using a **40/40/20 stratified sampling** strategy (40% High Complexity, 40% Low Complexity, and 20% Random) from the archival pool, this subset is used to establish numerical human-machine agreement for object and scene classification tasks.

This "Distillation" strategy allows us to index the entire 53,000 image archive using the YOLOv11 Student model, which was trained on the Tier 2 and Tier 3 data.

## 5. Citation

If you use this dataset or pipeline in your research, please cite the following thesis:
*Brhanu Atsbaha, "Object Detection and Relational Understanding in Historical Archives," Master's Thesis, University of Hildesheim, 2025.*
