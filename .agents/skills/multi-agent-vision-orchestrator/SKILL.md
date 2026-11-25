---
name: multi-agent-vision-orchestrator
description: Design, orchestrate, and audit decoupled 3-tier multi-agent visual analysis architectures combining spatial object detectors (YOLO), zero-shot classifiers (SigLIP), VLMs (LLaVA), grounded OCR (Kosmos-2.5), and LLM coordinators.
---

# Multi-Agent Vision Orchestrator & Architectural Skill

Use this skill when designing, building, or auditing multi-agent systems (MAS) for computer vision, visual-language curation, and archival indexing.

## Architectural Principles

### 1. Decoupled 3-Tier Execution Topology
Organize agents into strict, decoupled operational tiers to prevent mathematical circularity and chat token inflation:
- **Tier 1 (Validation Core)**: Local models (YOLOv11+SAM, SigLIP, LLaVA-OV, Kosmos-2.5) running offline to compute spatial grounding, VQA reasoning, and Scene-Aware Agreement ($SAA$).
- **Tier 2 (Historiographic Curation)**: Specialized downstream modules (Agent 2 Contradiction Detector, Agent 4 Cross-Archive Linker, Agent 5 Grounded OCR) consuming resolved post-validation metadata post-hoc.
- **Tier 3 (Frontier Selective Router)**: Commercial API coordinators (Gemini 2.5 Flash / Claude 3.5 Sonnet) invoked selectively for high-uncertainty scenes ($\sigma > \tau$).

### 2. Structured JSON Schema Protocol
- Abandon free-form unstructured text chat between agents.
- Enforce rigid numerical JSON communication schemas:
  ```json
  {
    "image_id": "COLIBRI_00542",
    "spatial_grounding": {"boxes": [[42, 110, 230, 410]], "labels": ["person"], "confidence": [0.89]},
    "scene_prior": {"category": "family", "confidence": 0.82},
    "vqa_reasoning": {"description": "A group portrait of a historical family.", "entities": ["person", "furniture"]},
    "consensus": {"saa_score": 0.865, "uncertainty": 0.135, "triage_flag": false}
  }
  ```

### 3. Generative Normalization & Archival Integrity
- Implement generative preprocessing (CycleGAN + Real-ESRGAN + Stable Diffusion Img2Img) to translate noisy out-of-distribution visual inputs into statistically modern VLM feature spaces.
- Enforce the **Axiom of Archival Integrity**: Restored images serve exclusively as ephemeral inference layers; primary published metadata anchors 100% back to original digitized scans.

### 4. Downstream Historiographic Enrichment
- **Agent 2 (Contradiction Detector)**: Compute explicit set-differences between VLM-mentioned nouns and spatial detector bounding boxes to detect hallucinations with high precision.
- **Agent 5 (Grounded OCR)**: Map bounding-box-aligned text tokens onto visual regions to corroborate scene labels.
