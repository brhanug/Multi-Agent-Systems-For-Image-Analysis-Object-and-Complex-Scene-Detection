# 🖼️ Historical Image Restoration & Analysis Dataset

## Overview

This dataset was created as part of the thesis: **"Visual Historian: Multi-Modal Restoration and Analysis of Historical Image Archives"**.

It comprises **12,110 historical images** (with ~300 domain-translated samples) processed through a comprehensive pipeline including:
1. **Restoration**: CycleGAN (Domain Translation) + Real-ESRGAN (Super-Resolution).
2. **Analysis**: Object Detection (YOLO, GroundingDINO, OWL-ViT) + Scene Classification (CLIP/SigLIP).
3. **Captioning**: Vision-Language Models (BLIP-2, LLaVA).

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

For a live demonstration of the project's capabilities, including the VQA interface and the Archive Metadata Explorer, please see the [**Demonstration Guide (DEMO.md)**](DEMO.md).

## 📊 Subset Stats (Diffusion/VQA Track)
- **Subset Size**: 231 images (`diffusion_restored/` workflow subset)
- **Formats**: `.jpg` (Original/CycleGAN), `.png` (Restored)
- **Annotations**:
    - Object Detection: YOLO format (`.txt`) + JSON
    - Captions: Natural language descriptions
    - Scene Types: Soft labeled categories (Family, Landscape, etc.)

---

## Final Dataset Structure

The `final_dataset/` directory provides a **unified, production-ready view** of all processed data and metadata. It uses symlinks to avoid duplication while maintaining a clean, hierarchical organization.

### Directory Overview

```
final_dataset/
├── images/          # All image variants (12,110 originals + processed versions)
├── labels/          # Multi-modal annotations and captions
└── metadata/        # Master manifest + analysis results
```

### Images (`images/`)
- **`original/`**: 12,110 images in 1,724 PPN-based directories (12MB)
- **`restored/`**: 12,110 Real-ESRGAN super-resolution images (symlink)
- **`cyclegan/`**: 302 domain-translated images (historical → modern, symlink)
- **`diffusion_restored/`**: 231 diffusion-enhanced images (symlink)
- **`visualizations/`**: 151 triptych collages for model comparison

### Labels (`labels/`)
- **`vlm_captions/blip2/`**: BLIP-2 generated captions (231 images, symlink)
- **`vlm_captions/llava_vqa.json`**: LLaVA-OneVision VQA responses (231 images, symlink)
- **`pseudo_labels/`**: spaCy-extracted noun labels (symlink)
- **`scene_labels/`**: CLIP + SigLIP scene classifications (symlink)
- **`yolo_v2_augmented/`**: YOLOv11 filtered training outputs (symlink)

### Metadata (`metadata/`)
- **`manifest.csv`**: ⭐ **Master index** linking all 12,110 images to their processed versions and metadata (1.7MB)
- **`vqa_binary_classification.json`**: VQA binary classification results (15 questions/image)
- **`srs_scores.json`**: Semantic Restoration Score (SRS) metrics
- **`agreement_scores/`**: Cross-model agreement analysis (symlink)

### Key Features

1. **PPN-Based Provenance**: Images organized by PPN (Pica Production Number) for traceability to source documents
2. **Symlink Architecture**: Avoids data duplication - only 12MB of actual files, rest are symlinks
3. **Manifest-Driven**: `manifest.csv` is the single source of truth for querying and analysis
4. **Multi-Modal**: Combines visual, textual, and semantic annotations

### Quick Start

```python
import pandas as pd

# Load master manifest
df = pd.read_csv('final_dataset/metadata/manifest.csv')

# Get all restored images
restored = df[df['restored_path'].notna()]

# Find images with captions
captioned = df[df['blip2_caption'].notna()]

# Query specific PPN
ppn_images = df[df['image_id'].str.startswith('PPN1752245350')]
```

For complete documentation, see [`final_dataset_structure.md`](.gemini/antigravity/brain/e374d846-d146-4eb2-94e0-7d71166174cc/final_dataset_structure.md).

---

## 🚀 Usage

### Loading the Manifest
The `metadata/manifest.csv` file is the entry point. It maps a unique `image_id` to all corresponding assets.

```python
import pandas as pd
df = pd.read_csv("metadata/manifest.csv")
print(df.head())
# Columns: image_id, original_path, restored_path, cyclegan_path, blip2_caption, scene_tags_clip
```

## 📜 Citation
[Placeholder for Citation]

## ⚖️ License
[Placeholder for License]
