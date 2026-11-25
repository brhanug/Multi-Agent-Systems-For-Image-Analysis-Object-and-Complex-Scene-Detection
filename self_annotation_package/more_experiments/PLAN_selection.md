# Is the learned scene-class combination's advantage an artefact of choosing it? (plan, 9 Nov 2026, before running)

The thesis adopted one joint mapping as its scene-class combination after E6 (PLAN_combination.md,
commit e9b630f) compared it on the same 200 redraws it is now reported on. Two checks, fixed before
they are run, both on the 992 pictures under one rule, the same 200 book-disjoint redraws (seed 20260926):

## S1: nested choice
In each redraw the combination is chosen using the training books only: five folds by book inside the
training books, each of {fixed 0.6/0.2/0.2, equal weights, learned weights (stacking), one joint mapping}
fitted on four folds and scored by balanced accuracy on the fifth; the variant with the highest mean is
refitted on all training books and scored on the test books. Reported: the chosen variant's balanced
accuracy and macro-F1 on the test books (mean and 2.5/97.5 percentiles), the difference to the fixed
weights per redraw, and how often each variant is chosen. Reading: if the nested figure is close to the
joint mapping's 0.544 and its difference to the fixed weights stays above zero over the redraws, the
gain is not an artefact of the choice.

## S2: regularisation of the joint mapping
The joint mapping uses the regression default C = 1. Repeated at C = 0.1 and C = 10, nothing else
changed. Reported: balanced accuracy, macro-F1, approved at alpha 0.05. Reading: a sensitivity check
only; C is not changed in the thesis whatever the result.
