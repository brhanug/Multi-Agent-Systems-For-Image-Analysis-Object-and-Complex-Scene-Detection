---
name: thesis-defense-audit
description: Perform a deep-dive methodological, statistical, and mathematical audit on academic thesis manuscripts and defense slide decks to identify examiner vulnerabilities, statistical under-powering, data leakage, and metric inconsistencies.
---

# Thesis Defense & Manuscript Audit Skill

Use this skill when auditing academic thesis documents, defense slide decks, or paper submissions for methodological rigor, statistical consistency, and defense readiness.

## Audit Workflow

### 1. Mathematical & Theoretical Consistency
- **Fixed-Point Convergence & Contractions**: Verify that any claims of convergence (e.g., Banach Fixed-Point Theorem) explicitly state continuity, metric space boundaries, and Lipschitz gradient bounds ($\gamma < 1$).
- **Shapley Coalitional Non-Circularity**: Ensure game-theoretic Shapley attributions ($\phi_i$) define coalitional players independently from global consensus targets to prevent mathematical circularity.
- **Conformal Risk Control**: Confirm non-conformity loss functions ($L_j(\lambda)$), risk bounds ($\alpha$), calibration sample sizes ($n_{\text{cal}}$), and test coverage bounds ($\mathcal{R}(\hat{\lambda}) \le \alpha$) are rigorously defined.

### 2. Experimental Rigor & Statistical Integrity
- **Proxy vs. Human Ground Truth**: Distinguish between synthetic proxy metrics (e.g., pseudo-label self-consistency mAP = 0.989) and human-audited expert ground truth (e.g., true human mAP = 0.1462).
- **Cohort Powering**: Check sample sizes across evaluation cohorts ($C_0 \dots C_k$). Flag small pilot cohorts ($n < 30$) as qualitative/feasibility studies rather than population proofs.
- **Metric Harmonization**: Cross-check numbers between Abstract, Introduction, Body Chapters, Figure Captions, and Presentation Slides for zero discrepancies.

### 3. Data Leakage & Evaluation Splitting
- Verify logistic calibration (Platt Scaling) and hyperparameter tuning are fitted strictly on training/calibration splits and evaluated on held-out test splits.
- Ensure cross-validation splits maintain strict independence without domain contamination.

### 4. Defense Q&A Vulnerability Identification
Generate targeted defense answers for top examiner traps:
- **Baseline Equivalence**: Explain why equal raw accuracy does not diminish value if the framework provides epistemic uncertainty quantification and error triage.
- **Metric Non-Correlation**: Clarify when metrics target high-level semantic ambiguity vs. micro-coordinate drawing jitter.
- **Generative Artifact Integrity**: Ensure generative restoration layer outputs are defined strictly as internal inference features, leaving primary catalog records unchanged.
