# 🖼️ Historical Image Restoration & Analysis — Multi-Agent Pipeline

> **Thesis**: *"Visual Historian: Multi-Modal Restoration and Analysis of Historical Image Archives"*

This repository contains the **dataset**, **pipeline scripts**, and **analysis tools** produced for the thesis. It comprises **12,110 historical images** (with ~300 domain-translated samples) processed through a multi-stage pipeline:

| Stage | Models |
|---|---|
| **Restoration** | CycleGAN (domain translation) + Real-ESRGAN (super-resolution) |
| **Object Detection** | YOLOv11, GroundingDINO, OWL-ViT |
| **Scene Classification** | CLIP / SigLIP |
| **Captioning / VQA** | BLIP-2, LLaVA-OneVision |
| **Multi-Agent Fusion** | Agreement-weighted scoring with HITL review flags |

---

## 📂 Directory Structure

```
.
├── config.yaml                   # Central pipeline configuration
├── configs/                      # Taxonomy and path overrides
├── scripts/
│   ├── core_pipeline/            # Main pipeline scripts
│   │   ├── pipeline_utils.py     # Shared loader utilities (NEW)
│   │   ├── run_multi_agent_validation.py
│   │   ├── run_agent_comparison.py
│   │   └── ...
│   └── analysis_viz/             # Visualisation & reporting scripts
├── src/                          # Model inference scripts
│   ├── owlvit_batch.py           # Batch OWL-ViT inference
│   ├── pseudo_label_generator.py # OWL-ViT CSV → YOLO labels
│   └── ...
├── notebooks/                    # Exploratory notebooks
├── human_baseline_gold_kit/      # 300-image human-labelled gold set
├── fact_check/                   # Hallucination & sensitivity tests
└── thesis_site/                  # Next.js thesis companion site

final_dataset/                    # (Generated — not in repo)
├── images/
│   ├── original/          # 12,110 source images (PPN/Volume structure)
│   ├── cyclegan/          # ~300 domain-translated images
│   ├── restored/          # Real-ESRGAN super-resolution images
│   ├── diffusion_restored/# Diffusion-enhanced subset (231 images)
│   └── visualizations/    # Triptych comparison collages
├── labels/
│   ├── yolo_v2_augmented/ # YOLOv11 detections (best model)
│   ├── vlm_captions/      # BLIP-2 captions + LLaVA VQA responses
│   ├── scene_labels/      # CLIP + SigLIP scene classifications
│   └── pseudo_labels/     # spaCy-extracted noun pseudo-labels
└── metadata/
    ├── manifest.csv        # ⭐ Master index (12,110 rows)
    ├── vqa_binary_classification.json
    ├── srs_scores.json     # Semantic Restoration Score per image
    └── agreement_scores/   # Cross-model agreement results
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.lock
```

Key packages: `torch`, `transformers`, `ultralytics`, `pandas`, `pyyaml`, `Pillow`, `tqdm`.

### 2. Configure paths

Edit `config.yaml` to point `project.base_dir` at your data root:

```yaml
project:
  base_dir: "/path/to/your/thesis_project"
```

### 3. Run the multi-agent validation

```bash
python scripts/core_pipeline/run_multi_agent_validation.py \
    --config config.yaml \
    --dataset-root final_dataset \
    --output-dir results/multi_agent
```

### 4. Compare agent contributions

```bash
python scripts/core_pipeline/run_agent_comparison.py \
    --config config.yaml \
    --dataset-root final_dataset \
    --output-dir results/multi_agent
```

### 5. Explore the dataset programmatically

```python
import pandas as pd

df = pd.read_csv("final_dataset/metadata/manifest.csv")

# All restored images
restored = df[df["restored_path"].notna()]

# Images with captions
captioned = df[df["blip2_caption"].notna()]

# Filter by archive PPN
ppn_images = df[df["image_id"].str.startswith("PPN1752245350")]
```

---

## 🖥️ Interactive Demos

See **[DEMO.md](DEMO.md)** for full instructions.

| Tool | Port | Purpose |
|---|---|---|
| VQA Interface | 7860 | Ask LLaVA-OneVision questions about any image |
| Archive Explorer | 7862 | Browse all 12,110 enriched images and metadata |

---

## 📊 Pipeline Overview

```
Raw Images (12,110)
      │
      ▼
  CycleGAN ──► Domain-Translated (~300)
      │
      ▼
  Real-ESRGAN ──► Super-Resolved Images
      │
      ├──► Object Detection (YOLO / GroundingDINO / OWL-ViT)
      ├──► Scene Classification (CLIP / SigLIP)
      ├──► Captioning (BLIP-2 / LLaVA VQA)
      └──► Multi-Agent Fusion ──► Realism Score + HITL Flags
```

Multi-agent weights (configurable in `config.yaml`):

| Agent | Default Weight |
|---|---|
| Object Detection | 0.25 |
| Cross-Model Agreement | 0.20 |
| Scene Classification | 0.15 |
| VLM / VQA | 0.15 |
| Image Restoration (SRS) | 0.15 |
| Document Grounding | 0.10 |

---

## 📁 Key Scripts

| Script | Purpose |
|---|---|
| `scripts/core_pipeline/run_multi_agent_validation.py` | Main validation: per-image realism scores + HITL flags |
| `scripts/core_pipeline/run_agent_comparison.py` | Monolithic vs coordinator fusion comparison |
| `scripts/core_pipeline/run_error_analysis.py` | Failure-mode taxonomy for high-disagreement images |
| `scripts/core_pipeline/pipeline_utils.py` | **Shared loader utilities** (new — used by all pipeline scripts) |
| `src/owlvit_batch.py` | OWL-ViT batch inference with GPU memory management |
| `src/pseudo_label_generator.py` | Convert OWL-ViT CSV detections → YOLO `.txt` labels |
| `scripts/analysis_viz/vqa_interface.py` | Gradio VQA demo |
| `scripts/analysis_viz/archive_visualizer.py` | Gradio archive explorer |

---

## 📜 Citation

If you use this dataset or pipeline in your work, please cite:

```bibtex
@mastersthesis{visualhistorian2025,
  author  = {Brhanu G.},
  title   = {Visual Historian: Multi-Modal Restoration and Analysis of Historical Image Archives},
  school  = {[University Name]},
  year    = {2025},
}
```

---

## ⚖️ License

This project is released for academic use. The underlying datasets (Colibri historical image archive) are subject to their respective institutional licenses. See individual data directories for provenance information.
