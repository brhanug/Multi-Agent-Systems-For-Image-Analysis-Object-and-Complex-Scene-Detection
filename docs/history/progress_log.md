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
| **Agent 0 & Dual Coordinator** | Upgrade YOLOv11 to YOLOv11-seg+SAM, Kosmos-2.5, SigLIP, and transition to Primary VLM + Critic LLM routing framework | Q2 2026 |

---

## 🎯 Next Immediate Actions (June 2026 — Phase 2 Architecture)

1. Define structured JSON payload schema between Agent 0 (YOLOv11-seg+SAM, Kosmos-2.5, SigLIP) and Dual Coordinator.
2. Implement YOLOv11-seg + SAM inference scripts to generate precise instance masks.
3. Set up Kosmos-2.5 processing for text-heavy archival images (structured markdown text).
4. Build SigLIP embedding-based zero-shot semantic categorization module.
5. Configure Primary VLM/LLM Prompt (semantic synthesizer) & Critic LLM Auditor routing framework.
6. Refactor MAS Core context injection (Geospatial, Demographics, Hallucination Critic) to consume upgraded Agent 0 inputs.

---

## 🧩 Updated Pipeline Summary

**CycleGAN → Diffusion Restoration → Agent 0 (YOLOv11-seg+SAM, Kosmos-2.5, SigLIP) → MAS Core (Geospatial, Demog, Critic) → Primary VLM/LLM Coordinator → Critic LLM Coordinator → Semantic Index / HITL Queue**

| Component | Role |
|------------|------|
| **CycleGAN** | Historical → modern translation |
| **Diffusion Restoration** | Enhances texture and fidelity |
| **YOLOv11-seg + SAM** | Precise pixel-perfect instance masking (objects) |
| **Kosmos-2.5** | Spatially-aware OCR & structured text-to-markdown |
| **SigLIP / CLIP** | Embedding-based scene/semantic classification |
| **MAS Context Core** | Geospatial, demographic, and historical context injection |
| **Primary VLM/LLM** | Text-centric semantic synthesizer (GPT-4o/Sonnet/Gemini Pro) |
| **Critic LLM** | Disagreement Auditor & Router (replaces hardcoded AWLF) |
| **Semantic Index** | High-confidence structured metadata target |
| **HITL Queue** | Human-in-the-loop review queue for audited conflicts |

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

---

**Log Update — April 25, 2026**

### **Defence Gap Analysis + Full Extended Evaluation Implementation**

#### Critical Fixes Applied
- **GitHub URL** corrected in Appendix (old `historical-image-analysis-thesis` → new repo)
- **RQ1 narrative reframed**: coordinator conservatism is intentional design, not a weakness. Wilcoxon W=14,443,727, p<0.001, Cohen's d=0.11 (negligible effect) → statistically significant but practically small difference.
- **Ablation anomaly explained** (document/restoration agents hurting fusion): documented as enrichment channels, not decision signals. Their proxy scores carry different semantic content than grounding agents.
- **"Latest Progress Update" section** rewritten as proper `Section 6: Extended Evaluation` chapter.

#### New Experiment Scripts Added
| Script | Purpose | Key Result |
|--------|---------|------------|
| `run_gold_simulation.py` | Consensus-of-consensus gold subset | 3,069 images (25.3%), both strategies F1=1.0 |
| `run_cross_fold_evaluation.py` | 5-fold stability | Δ=-0.0235 ± 0.0015 across folds; Cohen's d=0.11 |
| `run_rq2_disagreement_analysis.py` | Disagreement as error predictor | Pearson r=-0.7993 (strong inverse) |
| `run_complexity_analysis.py` | 5-bin complexity stratification | Fusion wins very_low bin by +0.176 (key RQ3 result) |
| `generate_thesis_figures.py` | 6 publication-ready figures | Agent dists, scatter, ROC, complexity bars, ablation, HITL |

#### Statistical Upgrades
- `run_statistical_report.py` upgraded with:
  - Wilcoxon signed-rank test (manual implementation, no scipy)
  - Cohen's d effect size labels (negligible/small/medium/large)
  - Pearson correlation matrix across all agents
  - `statistical_correlation_matrix.csv` added

#### New Output Artifacts (results/multi_agent/)
- `gold_simulation_subset.csv` + `gold_simulation_report.json`
- `cross_fold_results.csv` + `cross_fold_summary.json`
- `rq2_disagreement_analysis.json` + `rq2_pr_curve.csv` + `rq2_roc_curve.csv`
- `complexity_deep_analysis.csv` + `complexity_deep_summary.json`
- `statistical_correlation_matrix.csv`

#### Thesis Figures Generated (results/figures/)
- `thesis_agent_distributions.png`
- `thesis_mono_vs_fusion.png`
- `thesis_rq2_roc.png`
- `thesis_complexity_bars.png`
- `thesis_ablation_impact.png`
- `thesis_hitl_efficiency.png`

#### Committed & Pushed
- Commit: `dc2fc92` — pushed to `git@github.com:brhanug/Multi-Agent-Systems-For-Image-Analysis-Object-and-Complex-Scene-Detection.git`

---

**Log Update — June 3, 2026**

### **Transitioning to Upgraded Agent 0 + Dual-Coordinator Orchestration**
- **Decision**: Fully adopted the upgraded **Agent 0** instance-level understanding pipeline and modernized the **Dual-LLM Coordinator** orchestration.
- **Architectural Shift Details**:
  - **Segmentation Void Solved**: Swapped standard YOLOv11 detection with YOLOv11-seg + SAM, providing pixel-perfect instance masks.
  - **Text Ignorance Solved**: Integrated Kosmos-2.5 explicitly for spatially-aware text extraction and markdown formatting.
  - **Compute Efficiency**: Delegated semantic classification to SigLIP / CLIP embeddings instead of VLM queries.
  - **MAS Core Integration**: Re-routed MAS inputs (Geospatial, Demographics) to feed on isolated masks and structured Kosmos-2.5 markdown.
  - **Orchestration Modernization**: Moved away from hardcoded disagreement math (AWLF) to a text-centric Primary VLM/LLM Coordinator (e.g. GPT-4o / Claude 3.5 Sonnet / Gemini 1.5 Pro) paired with a Critic LLM Auditor (Claude 3.5 Haiku / Gemini 1.5 Pro).
- **Tasks Initiated**:
  - Define schema for the structured JSON payload output by Agent 0 and MAS core.
  - Script integration of YOLOv11-seg + SAM for mask extraction.
  - Set up Kosmos-2.5 pipeline for structured text extraction.
  - Design prompts for Primary VLM/LLM Coordinator and Critic LLM.
  - Implement the routing logic based on Critic LLM confidence.