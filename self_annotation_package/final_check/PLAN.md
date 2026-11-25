# Final check: does the corrected design keep its promise on new pictures? (plan fixed 4 Nov 2026)

Written and committed before any picture in `items.csv` was labelled. Nothing below is changed after
the labels are in; anything added later is reported as an extra analysis, not as the result.

## Why
Outside the 1,250-picture pool, the corrected design at line 0.5 and risk level 0.05 had about 15.5%
of pictures approved and wrong (355 pictures from Tasks E and A2, labelled by a person; in the pool
0.0498). Two likely causes: the pool's labels mix two rules for "scene", and the Task E/A2 pictures are
not a random draw from the archive.

## Pictures (`items.csv`, 704, shuffled, seed 20261105; the page does not show which set a picture is from)
- `random_archive`: 400 pictures drawn at random from the 10,860 archive pictures outside the pool
  (17 of them were also labelled on the quick-labels page; both labels are kept, and their agreement
  is reported).
- `scene_consistency`: the 304 pictures of `scene_consistency/candidates.csv` not in `verify_60.csv`.

## Rule
Scene = people, animals, a landscape or a room. No scene = sheet music, text, a cover, blank or marbled
paper, a border, an ornament, an initial without figures, a diagram, map or table, an emblem, coat of
arms or stamp (even with a figure), a single object, a plant study. Unsure only if one cannot tell.

## Labels for fitting and calibration (the pool)
The pool's labels, with each of the 364 scene-consistency pictures set to the person's label (these
304 plus the 60 of `scene_consistency/labels.csv`). "Unsure" keeps the old label. The model labels
(`claude_labels.csv`) are not used.

## Method (unchanged from the thesis)
Three agents (object, SigLIP scene with the "no scene" score, LLaVA), weights 0.6/0.2/0.2, conformal
risk control at alpha 0.05, 200 book-disjoint redraws of the pool (60/20/20), seed 20260926.

## Test
The 400 random pictures, "unsure" left out. Per redraw and in the mean over redraws:
- realised risk = share of all test pictures approved and wrong (the promise), share approved,
  share of approved pictures that are wrong, AUROC;
- at line 0.5 and at the fitted line;
- on all 400, and on those whose book is in neither the fitting nor the calibration part.
Uncertainty: 2.5/97.5 percentiles over redraws, and a 95% bootstrap interval over the test pictures
(2,000 resamples, seed 20261105) for the mean realised risk.

## Reading
- Promise kept: mean realised risk at most 0.05.
- Reported both ways: the same run with the pool's labels as used so far, so the effect of one rule
  is visible.
- Secondary (reported as extra): adding the 355 Task E/A2 pictures to the pool.
- Agreement between the person's 304 labels and the model labels for the same pictures is reported.
