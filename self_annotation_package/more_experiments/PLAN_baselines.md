# Baselines for the three tasks (plan, 9 Nov 2026, before running)

The class tasks are compared with each agent alone, the scene matcher zero-shot and the majority answer,
but not with the old system's own answer or with the standard single-model baseline of the field. These
are added, each on exactly the pictures, labels and redraws of the system's own results.

Scene classes (992 pictures, one rule, the 200 book-disjoint redraws of Task S):
- B-old, the deployed design's answer: "no scene" if its SAA < 0.5, otherwise the top of the scene
  matcher's forced choice over the five scene sentences (softmax). No training.
- B-llava-zs, the describer zero-shot: the class with the highest yes-strength, "no scene" if none reaches
  0.5. No training.
- B-probe, the linear probe: a multinomial logistic regression on the SigLIP image embedding (1,152
  numbers, L2-normalised), fitted on the training books, C chosen from {0.1, 1, 10} by three-fold book
  cross-validation inside the training books (balanced accuracy). The standard baseline for classifying
  with a frozen foundation model (Radford et al. 2021).
  Measures: balanced accuracy, macro-F1, accuracy; for B-probe also approval at alpha 0.05 (same CRC).
  Paired differences: the system (joint mapping) minus each baseline.
Objects (300 annotated pictures, the 200 redraws of Task O, 40/30/30):
- B-old, the deployed detector alone (its per-class box score, the object finder before retraining).
- B-probe, one binary logistic regression per class on the SigLIP embedding, C chosen as above (AUROC).
  Measures: mean AUROC over the eight classes, and for B-probe the missed-object promise at alpha 0.10
  (classes listed, objects missed). Paired: the system (mean of three) minus each baseline, mean AUROC.
Scene presence (1,250 pictures, labels as for the main results, the 200 redraws):
- B-probe, a binary logistic regression on the SigLIP embedding, C chosen as above; routed by confidence
  under CRC like the corrected design. Measures: AUROC, approval at alpha 0.05 and 0.10.
  Paired: the corrected design minus B-probe.
Reading: each baseline is reported whatever the result; a difference counts as established by the
thesis rule (range excludes zero).
