# 🧩 Thesis Progress Log — December 16, 2025

## ✅ Completed Stages

| Stage | Module | Status | Key Output |
|-------|---------|--------|-------------|
| 1️⃣ | **CycleGAN – Domain Translation** | ✅ Completed | `/pytorch-CycleGAN-and-pix2pix/results/hist2modern_v2/test_latest/images/` |
| 1.5️⃣ | **Diffusion Restoration – Post-Processing Enhancement** | ✅ Completed | `/thesis_project/results/diffusion_restored/` |
| 2️⃣ | **BLIP-2 – Caption Generation** | ✅ Completed | `/thesis_project/results/blip2_output/blip2_captions.json` |
| 3️⃣ | **spaCy – Pseudo-label Extraction** | ✅ Completed | `/thesis_project/results/pseudo_labels/` |
| 4️⃣ | **GroundingDINO – Phrase-Guided Detection** | ✅ Completed | `/thesis_project/results/groundingdino_v2/` |
| 5️⃣ | **OWL-ViT – Open-Vocabulary Detection** | ✅ Completed | `/thesis_project/results/owlvit_v2/` |
| 6️⃣ | **YOLOv8 – Pseudo-Supervised Fine-tuning** | ✅ Completed | `/thesis_project/runs/detect/optimized_v32/` |
| 7️⃣ | **YOLOv11 – Transformer-Enhanced Detection + Unified Dataset Export** | ✅ Completed | `/thesis_project/results/yolo11_final_run/exp_final_v3/` |
| 8️⃣ | **Extended Vision-Language Integration (CLIP + LLaVA)** | 🚧 In Progress | `/thesis_project/extensions/next_phase/` |

---

## ⚙️ Key Technical Updates

### **CycleGAN (Hist → Modern Translation)**
Trained with `--lambda_identity 0.5` and CLAHE histogram equalization → improved brightness and texture continuity.  
Generated **231** visually restored images.

### **Diffusion Restoration (Post-CycleGAN Enhancement)**
Model: *Image-to-Image Diffusion Refinement (Stable Diffusion v1.5 based)*.  
Applied to all CycleGAN outputs to enhance **texture fidelity**, **contrast**, and **fine details**.  
Generated **231** enhanced images saved in `/results/diffusion_restored/`.  
These became the **main visual source** for pseudo-label fusion and YOLOv11 fine-tuning.  
Significantly improved downstream detection performance for **OWL-ViT**, **GroundingDINO**, and **YOLOv11**.

### **BLIP-2 (Caption Generation)**
Model: `Salesforce/blip2-flan-t5-xl`.  
Produced captions for all 231 images (3–5 key nouns per caption).  
Example: *“A family walking with horses in a rural landscape.”* → extracted: `family`, `horses`, `landscape`.

### **spaCy (Noun-based Label Parser)**
Extracted structured object candidates from BLIP-2 text (POS = NN/NNS/NNP).  
Output stored per image under `/results/pseudo_labels/`.

### **GroundingDINO**
Phrase prompts: *“children playing”*, *“group of soldiers”*, *“family portrait”*, etc.  
Confidence ≥ 0.35 → **1,386** contextual detections saved in `/results/groundingdino_v2/`.

### **OWL-ViT (Zero-Shot Detection)**
Model: `google/owlvit-base-patch32` with confidence ≥ 0.10.  
Processed 231 images → 140 valid → 241 detections.  
Main classes: person (181), historical photo (30), woman (14), man (8), horse (7), family (1).

### **YOLOv8 (Pseudo-Labeled Fine-tuning)**
Model: `yolov8m.pt`, trained for 100 epochs (`lr0=0.001`, `lrf=0.01`, `mosaic=1.0`).  
Validation (`optimized_v32`) → **Precision 0.008**, **Recall 0.028**, **mAP@50 0.0047**.

### **YOLOv11 (Transformer-Enhanced Detection)**
Migrated pipeline to `yolo11m.pt`.  
Dataset: 231 entries, 4 classes (child, family, horse, person).  
Trained 100 epochs on GPU4 (RTX A6000).  
**Results:**  
- Precision 0.005  
- Recall 0.375 – 0.625 (varies by class)  
- mAP@50 0.021  
- mAP@50–95 0.002  
Moderate recall; low precision due to pseudo-label noise and limited data.  
Dataset alignment verified — all coordinates normalized.

---

## 📊 Current Outputs (GPU4)

| Component | Output Count | Folder |
|------------|--------------|--------|
| CycleGAN translated images | 231 | `results/hist2modern_v2/test_latest/images` |
| Diffusion-restored images | 231 | `results/diffusion_restored/` |
| BLIP-2 captions | 231 | `results/blip2_output/` |
| Pseudo-labels (spaCy) | 231 | `results/pseudo_labels/` |
| GroundingDINO detections | 1,386 | `results/groundingdino_v2/` |
| OWL-ViT detections | 241 | `results/owlvit_v2/` |
| YOLOv8 training run | 1 | `runs/detect/optimized_v32/` |
| YOLOv11 fine-tuning run | 1 | `results/yolo11_final_run/exp_final_v3/` |
| Unified dataset (v3) | 231 images + 231 labels | `results/dataset_export/yolo11_dataset_v3/` |

---

## 🔍 New Directions (Supervisor Feedback)

### 🆕 1. Refine YOLOv11 Training
- Apply confidence threshold ≥ 0.25 to pseudo-labels.  
- Introduce augmentations (mosaic, mixup, color jitter).  
- Compare YOLOv8 vs YOLOv11 precision/recall/mAP.

### 🆕 2. Unified Dataset + Zenodo Publication
Combine CycleGAN, **Diffusion Restoration**, BLIP-2, GroundingDINO, OWL-ViT, and YOLO outputs.  
Export as `.csv` + `.jsonl` for Zenodo release (Q1 2026).

### 🆕 3. Dual-Level Classification
- **Object-level:** person, horse, child, animal, building (≈ 10 frequent).  
- **Scene-level:** family, play, teaching, landscape, nature (≈ 5).  
Use CLIP or SigLIP zero-shot scene categorization from BLIP-2 captions.

### 🆕 4. Cross-Model Agreement
\[
Agreement(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]  
Compute overlaps among OWL-ViT, GroundingDINO, and YOLO outputs.  
Flag low-agreement images for manual review.

### 🆕 5. Gold Annotation Subset
Select 100–200 images with lowest agreement scores.  
Assign for manual annotation to assess pseudo-label accuracy.

### 🆕 6. VQA Integration (LLaVA-OneVision v1.5)
Generate semantic QA responses for validation:  
*“Is there a person?”*, *“Is this a teaching scene?”*, *“Outdoor or indoor?”*  
Compare LLaVA answers with YOLO/OWL-ViT detections for interpretability alignment.

---

## 🧠 Extended Future Work Roadmap

| Objective | Description | Target |
|------------|-------------|---------|
| **YOLOv11 Evaluation** | Benchmark vs YOLOv8 on pseudo-labeled data | Q1 2026 |
| **Zenodo Dataset Publication** | Release multi-modal dataset (captions + detections + metadata) | Q1 2026 |
| **Scene-Level CLIP Classification** | Add zero-shot scene reasoning | Q1–Q2 2026 |
| **Model Agreement Metrics** | Quantify multi-model overlap | Continuous |
| **Human Gold Subset** | Benchmark for label quality | Q2 2026 |
| **LLaVA-based Validation** | QA interpretability check | Q2 2026 |

---

## 🎯 Next Immediate Actions (Dec 16 – 23 2025)

1. Regenerate YOLOv11 labels (conf ≥ 0.25, normalized coordinates).  
2. Retrain YOLOv11 for 200 epochs with mosaic/mixup augmentation.  
3. Integrate CLIP Scene Classifier → 5 scene labels per image.  
4. Compute Cross-Model Agreement Matrix.  
5. Select divergent samples for Gold Annotation subset.  
6. Deploy LLaVA-OneVision QA module for validation.  
7. Prepare Zenodo metadata + dataset documentation.

---

## 🧩 Updated Pipeline Summary

**CycleGAN → Diffusion Restoration → GroundingDINO → OWL-ViT → BLIP-2 → spaCy → YOLOv8/YOLOv11 → CLIP Scene Classifier → LLaVA VQA → Zenodo Export**

| Component | Role |
|------------|------|
| **CycleGAN** | Historical → modern translation (enhancement) |
| **Diffusion Restoration** | Texture and detail refinement (Stable Diffusion) |
| **GroundingDINO / OWL-ViT** | Object & phrase grounding |
| **BLIP-2** | Caption generation & semantic context |
| **spaCy** | Noun extraction → pseudo-labels |
| **YOLOv8 / YOLOv11** | Pseudo-supervised fine-tuning |
| **CLIP** | Scene-level zero-shot classification |
| **LLaVA-OneVision** | Textual reasoning & VQA validation |
| **Zenodo Export** | Final dataset for reproducibility |

---

**Log Update — April 21, 2026**

### **Professor-Requested Evaluation Reframe: "Whole Pipeline as One Agent"**
- Added a monolithic baseline (`monolithic_pipeline_agent`) to represent the full workflow as a single opaque decision score.
- Kept the coordinator fusion (`comparison_fusion_score`) as the disagreement-aware multi-agent comparator.
- Updated and re-ran:
  - `scripts/core_pipeline/run_agent_comparison.py`
  - `scripts/core_pipeline/run_research_evaluation.py`
  - `scripts/core_pipeline/run_statistical_report.py`
- Refreshed artifacts in `results/multi_agent/`:
  - `agent_comparison_scores.csv`
  - `agent_comparison_summary.json`
  - `research_baseline_summary.csv`
  - `research_ablation_summary.csv`
  - `research_hitl_efficiency.csv`
  - `research_evaluation_summary.json`
  - `statistical_ci_summary.csv`
  - `statistical_pairwise_deltas.csv`
  - `statistical_report_summary.json`
- Current snapshot (12,110 images):
  - `monolithic_pipeline_agent` mean: `0.5867`
  - `comparison_fusion_score` mean: `0.5632`
  - provenance: `mixed` (measured + proxy)