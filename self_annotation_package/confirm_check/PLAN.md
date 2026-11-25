# Confirmation check: does calibrating on archive pictures keep the promise? (plan fixed 5 Nov 2026)

Written and committed before any picture in `items.csv` was labelled. Nothing below changes after the
labels are in; anything added later is reported as an extra analysis.

## Why
The final check (`final_check/PLAN.md`, commit 51428dc) found that a cut-off calibrated on the labelled
pool does not keep the promise on random archive pictures. An analysis added after that test, and so
not planned, found that calibrating on half of the random archive pictures keeps it (realised risk
0.048, 81% approved under one rule). This check tests that remedy on pictures nobody has used.

## Pictures (`items.csv`)
300 pictures drawn at random (seed 20261106) from the archive pictures outside the 1,250-picture pool
and outside the 400 of the final check (10,460 candidates). Four were also labelled on the quick-labels
page; both labels are kept.

## Rule
The same as the final check: scene = people, animals, a landscape or a room; no scene = sheet music,
text, a cover, blank or marbled paper, a border, an ornament, an initial without figures, a diagram,
map or table, an emblem, coat of arms or stamp (even with a figure), a single object, a plant study.
Unsure only if one cannot tell; unsure pictures are left out and counted.

## Method (fixed)
- Fit: the three agents' mappings and the 0.6/0.2/0.2 combination on the whole pool, with the pool's
  labels under one rule (the person's labels on the 364 scene-consistency pictures).
- Calibrate: conformal risk control at alpha 0.05 on the 399 labelled random pictures of the final
  check (the one unsure picture left out).
- Test: the 300 new pictures, once.
- Primary: line 0.5. Also reported: the fitted line, the pool's labels as used so far, and alpha 0.025.

## Reading
The promise bounds the average over calibration sets; a single calibration set exceeds the level in
about 40% of cases even when the promise holds (thesis, per-batch risk). So:
- Primary figure: the mean realised risk on the 300 over 500 bootstrap resamples of the calibration
  pictures (by book, seed 20261106). Promise kept: that mean is at most 0.05.
- Also reported: the realised risk with the calibration set as drawn, with a 95% bootstrap interval
  over the test pictures, and the share approved.
- A single-calibration risk above 0.05 whose interval includes 0.05 is read as consistent with the
  promise; an interval entirely above 0.05 as a failure.
