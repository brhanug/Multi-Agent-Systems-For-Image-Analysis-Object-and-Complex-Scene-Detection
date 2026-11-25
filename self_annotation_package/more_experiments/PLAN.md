# Five further experiments (plan fixed 9 Nov 2026, before any of them was run)

No new labels. Everything uses labels that already exist. Nothing below changes after the results are
known; anything added later is reported as an extra analysis.

Common setting unless stated: the corrected design (three agents, mappings fitted on training books,
0.6/0.2/0.2, uncertainty = closeness to the line 0.5), pool labels under one rule, 200 book-disjoint
redraws 60/20/20 (seed 20260926). "Archive pictures" = the 693 labelled random archive pictures
(399 final check + 294 confirmation check).

## E1. Weighted calibration without archive labels
Question: can the pool calibration be repaired for the archive using only unlabelled archive pictures?
Under one rule the remaining difference is that the pool is not a random sample (covariate shift).
- Weights: a logistic regression (C = 1) separates the pool's 1,250 pictures from the unlabelled archive
  pictures (all scored pictures outside the pool and outside the 700 test pictures), on the agents'
  reports (22 numbers). Weight w(x) = P(archive | x) / P(pool | x), rescaled by the class sizes.
- Weighted conformal risk control (Tibshirani et al. 2019, applied to the risk): for each archive test
  picture j, the largest cut-off with (sum_i w_i L_i(lambda) + w_j) / (sum_i w_i + w_j) <= alpha over the
  pool calibration pictures i; approve j if its uncertainty is below its own cut-off.
- Measured on the 693 archive pictures, mean over redraws: realised risk and share approved, at
  alpha 0.05 and 0.10, against the same design without weights. Also the effective sample size of the
  weighted calibration set.
- Reading: the weighting works if the mean realised risk at alpha 0.05 is at most 0.05.

## E2. A promise per calibration set (Learn then Test)
Question: can the promise hold for the calibration set in hand, not only on average?
- Learn then Test (Angelopoulos et al. 2021) with delta = 0.10: for each cut-off on the grid, from the
  most cautious upward, the binomial p-value of "risk > alpha" (the loss is 0/1); fixed-sequence testing
  stops at the first cut-off not rejected at delta; the last rejected cut-off is used.
- On the pool (labels as used for the headline, as in the per-batch analysis) over the 200 redraws:
  share of redraws whose test risk exceeds alpha (target at most 10%) and share approved, against
  conformal risk control, at alpha 0.05 and 0.10.
- On the archive: calibrate on the 399 final-check pictures, test on the 294 confirmation pictures, one
  rule; realised risk and share approved at alpha 0.05.

## E3. Prediction sets with a promise per scene class
Question: can short lists cover the rare scene classes too?
- Task S as planned (six answers, 992 pictures), with the planned and with class-balanced mappings.
- Class-conditional (Mondrian) sets: for each class k, the calibration pictures of class k give a
  threshold on p_k at level alpha = 0.10; k is in the set if p_k reaches it (always if the class has
  fewer than 9 calibration pictures). The top class is always included.
- Measured: coverage per class, overall coverage, mean set size, against the sets of Task S.

## E4. One model as the system
Question: does a design with the describer (LLaVA) alone hold up as well as the combination?
- First level of the scene decision, describer alone mapped to a probability of a scene, same protocol.
- On the archive: fitted and calibrated on the pool (as the final check), and calibrated on the 399
  final-check pictures and tested on the 294 (as the confirmation check): realised risk and approval at
  alpha 0.05, beside the combination.

## E5. Does restoration help?
Question: the corpus was restored with Real-ESRGAN before scoring; is that better than the raw scans?
- The scene matcher (SigLIP so400m, the same five scene prompts and the same seven "no scene" prompts)
  scores the original scans of the 992 Task S pictures and the 693 archive pictures. A sanity check
  recomputes the restored scores for 50 pictures and requires them to match the stored ones.
- Compared, restored against original, paired over the 200 redraws of Task S: the scene matcher alone
  (macro-F1, balanced accuracy) and zero-shot; the first level (AUROC of scene presence, scene matcher
  alone); and the AUROC on the 693 archive pictures with mappings fitted on the pool.
- Only the scene matcher is re-scored; the describer and the object finder are not (GPU time). The
  thesis states this limit.
