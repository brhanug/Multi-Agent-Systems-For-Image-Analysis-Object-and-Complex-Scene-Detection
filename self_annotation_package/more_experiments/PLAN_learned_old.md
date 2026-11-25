# The old design with every variable learned (plan, 9 Nov 2026, before running)

The SAA weights, the SAA threshold and the triage weight were each tested separately. This learns all of
them together, to give the strongest version the old design can reach, on the 1,250-picture pool, the
labels of the main results and the same 200 book-disjoint redraws (seed 20260926). Everything is fitted on
the training books only; the cut-off is set by conformal risk control on the calibration books.

- Learned SAA: a logistic regression on the deployed signals (object, scene, VLM; and a second variant with
  all four including Agreement); the call is scene if its probability >= 0.5, so weights and threshold
  are both learned.
- Learned triage, two variants: (a) the 0.6/0.4 form with the weight chosen on the training books
  (w in 0, 0.1, ..., 1, highest error AUROC); (b) a learned error model: a logistic regression predicting
  the learned SAA's errors from the four signals, their spread, their mean and |p - 0.5|, fitted on
  out-of-fold errors inside the training books (five folds by book); its predicted error probability is
  the routing score.
Reported: scene AUROC, approval and realised risk at alpha 0.05 and 0.10, beside the deployed old design
(fixed SAA, triage at w = 0.6) and the corrected design; paired differences to the corrected design.
