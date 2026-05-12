# Historical Image Multi-Agent Analysis Dataset

## Overview
This dataset was created as part of the thesis: "Multi-Agent System for Image Analysis: Object and Complex Scene Detection".

It comprises 12,110 historical images processed through a comprehensive **6-Agent System**:

- **Agent 0 (Foundational CV Pipeline)**: Generative Restoration (CycleGAN, Real-ESRGAN, Stable Diffusion), Object Grounding (GroundingDINO, Florence-2, YOLOv11), Scene Classification (CLIP/SigLIP), and text extraction (Kosmos-2.5).
- **Agent 1 (Temporal Historian)**: Tracks object-class drift across institutional PPN decades.
- **Agent 2 (Retrieval Agent)**: RAG semantic search across archival imagery.
- **Agent 3 (Hallucination Critic)**: Evaluates conflict between VQA semantics and spatial bounding boxes.
- **Agent 4 (Demographic Profiler)**: Computes social composition scores (persons/children/clothing).
- **Agent 5 (Geospatial Analyst)**: Classifies images on an urban/rural/institutional spectrum.

A central **Coordinator** fuses these 6 agents, routing high-uncertainty images to a Human-in-the-Loop review queue.

## 📂 Directory Structure
```
final_dataset/
├── images/
│   ├── original/          # Source historical images (Structured by PPN/Volume ID)
│   ├── cyclegan/          # Domain-translated images (Flat structure, subset ~300)
│   ├── restored/          # High-resolution restored images (SSPN Structure - PNG)
│   ├── diffusion_restored/# Diffusion-enhanced restored images (Flat - JPG)
│   └── visualizations/    # Analytical visualizations (Agreement Collages etc.)
├── labels/
│   ├── yolo_v2_augmented/ # YOLOv11 detections (Best performing model)
│   ├── vlm_captions/      # BLIP-2 captions and LLaVA VQA responses
│   ├── scene_labels/      # Scene classification (JSON)
│   └── pseudo_labels/     # Initial pseudo-labels extracted via spaCy
└── metadata/
    ├── manifest.csv       # Master index linking all files per image ID
    └── agreement_scores/  # Cross-model agreement analysis results
```

## 🚀 Interactive Demos
For a live demonstration of the project's capabilities, including the VQA interface and the Archive Metadata Explorer, please see the Demonstration Guide (`DEMO.md`).

## 📊 Subset Stats (Diffusion/VQA Track)
- **Subset Size**: 231 images (`diffusion_restored/` workflow subset)
- **Formats**: `.jpg` (Original/CycleGAN), `.png` (Restored)
- **Annotations**:
  - **Object Detection**: YOLO format (`.txt`) + JSON
  - **Captions**: Natural language descriptions
  - **Scene Types**: Soft labeled categories (Family, Landscape, etc.)

## Final Dataset Structure
The `final_dataset/` directory provides a unified, production-ready view of all processed data and metadata. It uses symlinks to avoid duplication while maintaining a clean, hierarchical organization.

### Directory Overview
```
final_dataset/
├── images/          # All image variants (12,110 originals + processed versions)
├── labels/          # Multi-modal annotations and captions
└── metadata/        # Master manifest + analysis results
```

### Images (`images/`)
- `original/`: 12,110 images in 1,724 PPN-based directories (12MB)
- `restored/`: 12,110 Real-ESRGAN super-resolution images (symlink)
- `cyclegan/`: 302 domain-translated images (historical → modern, symlink)
- `diffusion_restored/`: 231 diffusion-enhanced images (symlink)
- `visualizations/`: 151 triptych collages for model comparison

### Labels (`labels/`)
- `vlm_captions/blip2/`: BLIP-2 generated captions (231 images, symlink)
- `vlm_captions/llava_vqa.json`: LLaVA-OneVision VQA responses (231 images, symlink)
- `pseudo_labels/`: spaCy-extracted noun labels (symlink)
- `scene_labels/`: CLIP + SigLIP scene classifications (symlink)
- `yolo_v2_augmented/`: YOLOv11 filtered training outputs (symlink)

### Metadata (`metadata/`)
- `manifest.csv`: ⭐ Master index linking all 12,110 images to their processed versions and metadata (1.7MB)
- `vqa_binary_classification.json`: VQA binary classification results (15 questions/image)
- `srs_scores.json`: Semantic Restoration Score (SRS) metrics
- `agreement_scores/`: Cross-model agreement analysis (symlink)

## Key Features
- **PPN-Based Provenance**: Images organized by PPN (Pica Production Number) for traceability to source documents
- **Symlink Architecture**: Avoids data duplication - only 12MB of actual files, rest are symlinks
- **Manifest-Driven**: `manifest.csv` is the single source of truth for querying and analysis
- **Multi-Modal**: Combines visual, textual, and semantic annotations



