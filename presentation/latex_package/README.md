# LaTeX source package

Self-contained sources for the thesis, the defense deck and the handout,
with every referenced image included.

## Layout

    presentation/
      thesis_v3_updated.tex     the report
      defense_slides.tex        the defense deck (Beamer)
      defense_handout.tex       the 2-page committee handout
      references.bib            bibliography (shared)
    latex_assets/
      failure_gallery.tex       \input by the report's appendix
      *.png, *.jpg              all 24 referenced figures

The directory layout matters: the .tex files look for images in
`../latex_assets/`, so `latex_assets/` must stay a **sibling** of
`presentation/`, not a subfolder of it.

## Building

Compile from inside `presentation/`. Any of these work:

    cd presentation
    tectonic -X compile thesis_v3_updated.tex        # one pass, resolves citations itself
    tectonic -X compile defense_slides.tex
    tectonic -X compile defense_handout.tex

or with a standard TeX Live installation:

    cd presentation
    pdflatex thesis_v3_updated
    bibtex   thesis_v3_updated
    pdflatex thesis_v3_updated
    pdflatex thesis_v3_updated                       # third pass settles cross-references

The deck and the handout need no bibtex pass; the deck carries its
bibliography inline and the handout has no citations.

## Requirements

TeX Live 2021 or later (or tectonic). Packages used beyond a standard
install: `beamer` with the Metropolis theme, `tikz` + `pgfplots`,
`booktabs`, `subcaption`, `seqsplit`, `algorithm`/`algpseudocode`,
`hyperref`, `xcolor`. Tectonic downloads these automatically.

## Expected output

    thesis_v3_updated.pdf   100 pages
    defense_slides.pdf       51 pages
    defense_handout.pdf       2 pages
