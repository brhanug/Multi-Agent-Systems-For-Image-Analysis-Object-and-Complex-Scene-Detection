# Pre-registration: does class-level uncertainty beat the scalar version? (C4)

Written 19 September 2026, before any C4 analysis is run. The commit that adds this file is the
time stamp. Nothing below may be changed after C4 results exist without a dated note saying what
changed and why.

## Question

RQ1 was answered No on a scalar disagreement signal that collapses to eight states. A per-class
signal is a different object. Does uncertainty built at the image-and-class level detect the
pipeline's errors better than plain single-model confidence?

## Signals compared

1. Per-class disagreement: for each picture, the mean over classes of the spread of the three
   systems' calibrated per-class scores (detector, SigLIP, LLaVA).
2. Per-class entropy: for each picture, the mean over classes of the binary entropy of the fused
   per-class probability.
3. Global disagreement: the same spread computed on one score per system (its maximum over classes).
4. The deployed scalar signal, unchanged.
5. Plain confidence: one minus the fused scene probability's distance from the decision boundary
   (the baseline RQ1 was lost to).

Calibration of each system's per-class scores is fitted on the train partition only.

## Where it is evaluated

- **A. The 300 independently annotated pictures** (authoritative). Error = the pipeline's scene
  decision disagrees with the rater. The 136 pictures in the detector's training split are
  reported separately and the headline uses the other 164 plus the full 300.
- **B. The held-out test partition** of the scene pool, as it stands when C4 runs (currently 249
  pictures; 307 if the expanded pool has replaced it by then). Error defined as in the thesis.

## Metric and decision rule

ROC-AUC of each signal for detecting errors, and the paired difference between signal 1 and
signal 5 with a DeLong 95% confidence interval.

- If on set A the interval for (1 − 5) lies entirely above zero: **RQ1 becomes a qualified yes** —
  class-level disagreement adds information that confidence lacks.
- If it contains zero or lies below: **RQ1 stays no**, now at two levels of granularity.
- Set B is supporting evidence; it cannot overturn a result on set A.

Either outcome is reported in the thesis in full, including all five signals, whatever their order.

## What would make the test invalid

- Any calibration or threshold fitted on set A or on the test partition.
- Choosing among several disagreement definitions after seeing results: signal 1 is the one tested;
  the others are reported, not substituted.

## Amendment, 19 September 2026 (before any C4 analysis)

The calibration rule above cannot be carried out as written: per-class calibration needs object
labels, and the train partition of the scene pool has none — object labels exist only on the 300
annotated pictures (set A). The rule is therefore replaced by:

- **Set A:** each system's per-class calibration is cross-fitted on the 300 — five folds, fitted on
  four, applied to the fifth — so no picture is scored by a calibration that saw it.
- **Set B:** calibration is fitted on the 300 minus the 37 of them that fall in the test partition.

Nothing else changes. No C4 analysis had been run when this amendment was written.
