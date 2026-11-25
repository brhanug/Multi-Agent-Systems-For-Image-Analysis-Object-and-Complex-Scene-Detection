# Workspace Customizations & Agent Guidelines

This workspace is configured with four project-level agent skills for multi-agent visual analysis, uncertainty calibration, LaTeX manuscript compilation, and academic defense auditing.

## Available Workspace Skills

1. **`thesis-defense-audit`** (`.agents/skills/thesis-defense-audit/SKILL.md`):
   - Audits manuscripts and Beamer slides for mathematical consistency, statistical under-powering, data leakage, and examiner trap defense.

2. **`latex-manuscript-compiler`** (`.agents/skills/latex-manuscript-compiler/SKILL.md`):
   - Handles multi-file LaTeX compilation using Tectonic/pdfLaTeX, automatic table scaling (`adjustbox`), 16:9 Beamer layouts, and package error filtering.

3. **`epistemic-uncertainty-evaluator`** (`.agents/skills/epistemic-uncertainty-evaluator/SKILL.md`):
   - Evaluates Scene-Aware Agreement ($SAA$) disagreement, Platt logistic calibration ($ECE < 0.001$), and distribution-free Conformal Risk Control bounds.

4. **`multi-agent-vision-orchestrator`** (`.agents/skills/multi-agent-vision-orchestrator/SKILL.md`):
   - Orchestrates the 3-Tier Decoupled Multi-Agent System (Tier 1 Validation Core, Tier 2 Curation Agents, Tier 3 Frontier Selective Router).

## Enforcement Policy

- Always verify LaTeX sources with `latex-manuscript-compiler` prior to pdf compilation.
- Use `epistemic-uncertainty-evaluator` when auditing ECE, Platt scaling, or Conformal Risk Control metrics.
- Execute `thesis-defense-audit` when auditing thesis reports or defense slides against examiner vulnerabilities.
