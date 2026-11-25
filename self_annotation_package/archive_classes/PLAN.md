# The scene classes on random archive pictures (plan fixed 8 Nov 2026, before any labelling)

Written and committed with `items.csv` before any picture in it was labelled. Nothing below changes
after the labels are in; anything added later is reported as an extra analysis.

## Why
The scene-class task (class_tasks/PLAN.md, commit d9a7fa1) was measured inside the labelled pool. The
first level of the scene decision was tested on random archive pictures (final_check, confirm_check)
and did not carry over from the pool without archive calibration. This tests whether the scene-class
results carry over to random archive pictures.

## Pictures (`items.csv`, 444, shuffled with seed 20261108)
- 247 pictures of the final check and 157 of the confirmation check that the author labelled
  "scene" (404 random archive pictures);
- 40 hidden pool pictures with a known expert class (teaching 6, family 9, playing 8, landscape 9,
  drawing 8), to measure how far the author's reading of the classes matches the expert's.
`items_with_source.csv` says which is which; the labelling page does not show it.

## Labels
One of: teaching, family, playing, landscape, drawing, no scene, unsure. The expert left no written
definitions, so the page gives the author's reading of the class names:
- teaching: a lesson or instruction (a teacher with pupils, a school, someone being taught or read to);
- family: family life (parents with children, people at home together);
- playing: play, games or sport;
- landscape: outdoor scenery is the main subject, people small or absent;
- drawing: any other scene with people, animals or a room;
- no scene: on a second look, not a scene under the scene rule;
- unsure: cannot tell.
If two classes fit, the one describing the main subject; if still tied, teaching, playing or family
before landscape, and any of these before drawing.

## Method (fixed)
The 693 labelled random archive pictures (the 404 above with their new class, the 289 the author
labelled "no scene"; "no scene" and "unsure" given on this page are applied, unsure left out).
- Fit: the mappings of Task S on the whole pool of 992 pictures under one rule, planned (unweighted)
  mappings; class-balanced mappings as the second variant. Combination 0.6/0.2/0.2; describer alone
  reported beside it.
- Calibrate on the final-check pictures (399), test on the confirmation-check pictures (294), as in
  the confirmation check.
- Measures on the test pictures: macro-F1, balanced accuracy, recall and precision per class; at
  alpha 0.05 the share approved, the realised risk and the approvals by answer given; prediction sets
  at alpha 0.10 with coverage overall and per class.
- Also on all 693 (no risk control): macro-F1, balanced accuracy and recall per class.
- The 40 controls: Cohen's kappa between the author's and the expert's class, over the five classes.

## Reading
A class is reported only if the test pictures hold at least 5 of it; teaching is expected to be rare.
The pool results (Section "The Five Scene Classes") carry over if the archive figures lie within the
spread over the pool's redraws. The control kappa says how much of any difference may be a different
reading of the classes rather than a different archive.

## Amendment (9 Nov 2026, before any archive scene-class result was computed)
At the time of this amendment 51 of the 444 pictures were labelled and no analysis of them had been
written or run. Since the plan was fixed, the planned comparison E6 (more_experiments/PLAN_combination.md,
commit e9b630f) showed that a learned combination is clearly better for the scene classes than
0.6/0.2/0.2 (balanced accuracy 0.544 against 0.401 on the pool), and the thesis now uses one joint
mapping (a single multinomial regression on all 27 agent reports, fitted on the pool) as the system's
scene-class combination. The archive test therefore runs:
- primary: the joint mapping, planned (unweighted) fit; class-balanced fit as the second variant;
- the originally planned 0.6/0.2/0.2, both fits, reported beside it unchanged;
- the describer alone, as before.
Everything else above (pictures, calibration, test, measures, controls, reading) is unchanged.
