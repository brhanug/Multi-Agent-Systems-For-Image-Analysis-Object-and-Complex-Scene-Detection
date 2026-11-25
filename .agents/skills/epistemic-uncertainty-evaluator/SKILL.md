---
name: epistemic-uncertainty-evaluator
description: Evaluate epistemic uncertainty, inter-agent model disagreement, Expected Calibration Error (ECE), Platt scaling, and distribution-free Conformal Risk Control (CRC) bounds for computer vision and VLM pipelines.
---

# Epistemic Uncertainty & Conformal Calibration Evaluator Skill

Use this skill when evaluating uncertainty estimation, multi-model consensus disagreement, calibration metrics (ECE), or distribution-free risk control algorithms for computer vision and multimodal pipelines.

## Evaluation Framework

### 1. Scene-Aware Agreement (SAA) Disagreement Engine
Formulate inter-agent model disagreement between spatial grounders ($A^{\text{object}}$), zero-shot scene classifiers ($A^{\text{scene}}$), visual-language reasoners ($A^{\text{vlm}}$), and grounded OCR ($A^{\text{ocr}}$):
$$SAA_i = \phi_o A_i^{\text{object}} + \phi_s A_i^{\text{scene}} + \phi_v A_i^{\text{vlm}} + \phi_r A_i^{\text{ocr}}$$
$$U_{\text{SAA}, i} = 1 - SAA_i$$

### 2. Error Prediction ROC-AUC Benchmark
- Evaluate raw disagreement uncertainty ($U_{\text{SAA}}$) against single-model temperature-scaled entropy baselines.
- Compute ROC-AUC on expert-annotated gold reference splits to verify error detection discrimination.

### 3. Probabilistic Platt Calibration (Logistic Scaling)
- Map uncalibrated consensus scores $S_{\text{fusion}}$ to posterior error probabilities:
  $$P(y=1 | S_{\text{fusion}}) = \frac{1}{1 + \exp(A \cdot S_{\text{fusion}} + B)}$$
- Measure pre- and post-calibration Expected Calibration Error (ECE) across 10 uniform probability bins ($M=10$):
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- Target: Collapse ECE from $>0.20 \rightarrow < 0.001$.

### 4. Distribution-Free Conformal Risk Control (CRC)
- Calibrate non-conformity loss threshold $\hat{\lambda}$ on calibration split $n_{\text{cal}}$ at target risk bound $\alpha$:
  $$\mathcal{R}(\hat{\lambda}) = \mathbb{E}\left[ \mathbb{I}(\text{sign}(S_{\text{fusion}} \ge \hat{\lambda}) \neq y_{\text{expert}}) \right] \le \alpha$$
- Evaluate empirical test risk $\mathcal{R}_{\text{test}}(\hat{\lambda})$ on held-out test split $n_{\text{test}}$ to verify bound satisfaction.
- Quantify human review workload reduction (\%) and error recall (\%) under the certified threshold.
