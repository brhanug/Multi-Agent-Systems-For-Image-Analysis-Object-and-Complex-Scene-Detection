---
name: latex-manuscript-compiler
description: Compile, validate, and optimize complex multi-file LaTeX reports, Beamer slides, TikZ diagrams, and bibliography files using Tectonic/pdfLaTeX with zero errors and strict page-budget constraints.
---

# LaTeX Manuscript Compiler & Visual Optimizer Skill

Use this skill when editing, compiling, and formatting complex LaTeX reports, thesis documents, or Beamer presentation decks.

## Execution Rules

### 1. Preamble & Package Safeguards
- Always load `adjustbox`, `silence`, `booktabs`, `caption`, `subcaption`, `graphicx`, and `hyperref` with proper package ordering.
- Silence non-critical PDF bookmark warnings using:
  ```latex
  \usepackage{silence}
  \WarningFilter{hyperref}{Token not allowed in a PDF string}
  \WarningFilter{hyperref}{Difference (2) between bookmark levels}
  ```
- Implement image safe-loading macros (`\safeincludegraphics`) to gracefully handle unreadable figure paths during iterative builds.

### 2. Table & Figure Layout Optimization
- Wrap wide LaTeX tables in auto-scaling boxes:
  ```latex
  \begin{adjustbox}{max width=\linewidth}
  \begin{tabular}{...}
  ...
  \end{tabular}
  \end{adjustbox}
  ```
- Escape special characters (`&`, `%`, `_`, `$`, `#`) outside of tabular environments.
- Ensure figure captions reproduce the exact numerical metrics reported in the body text.

### 3. Beamer Presentation Best Practices
- Use $16:9$ widescreen aspect ratio (`\documentclass[aspectratio=169,11pt]{beamer}`).
- Format slide content in 2-column layouts (`\begin{columns}[T]`) with left text/table blocks and right figure blocks.
- Build clean TikZ pipeline diagrams using layered node styling (`positioning, shapes.geometric, arrows.meta, calc, fit, backgrounds`).

### 4. Compilation Workflow
- Execute non-interactive builds via Tectonic or pdfLaTeX:
  ```bash
  tectonic document.tex --keep-intermediates
  ```
- Inspect output logs for unescaped characters, missing fonts, or overfull `\hbox` warnings.
