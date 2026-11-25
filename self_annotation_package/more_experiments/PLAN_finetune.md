# Plan: does training SigLIP and LLaVA-OneVision on the archive help? (fixed 25 Nov 2026, before any run)

## Question
The scene matcher (SigLIP so400m-patch14-384) and the describer (LLaVA-OneVision 7B) are used zero-shot:
neither was trained on the archive; only the small mappings on their outputs are learned. Does
fine-tuning the two models themselves on the archive's labelled pictures improve the corrected design?

## Labels and pictures
- Training targets: the scene-class labels ("one rule") of the 992 pictures of Section 6.7
  (teaching, family, playing, landscape, drawing, none).
- Pictures scored: all 1,250 pictures of the labelled pool (the 992 are among them).
- Objects are out of scope (hand boxes exist for 300 pictures only); the object finder is unchanged.

## Cross-fitting by book (no picture is scored by a model that saw its book)
The 550 books of the pool are split once into 5 folds (seed 20261125). For each fold k, a model is
trained on the class-labelled pictures of the other four folds and scores the pictures of fold k.
Every picture thus has an out-of-fold score. Fine-tuning cannot be repeated for each of the 200
redraws; the downstream pipeline (mappings, combination, conformal risk control) then runs on these
scores over the same 200 book-disjoint redraws as the thesis. A test picture's score comes from a model
that never saw its book, but that model may have been trained on other books of the same redraw's
calibration or test part; this is reported as a limitation.

## Training (fixed before running)
- SigLIP: text features of the 12 sentences (5 scene, 7 "not a scene") frozen; the last 4 vision blocks,
  the vision head and the logit scale and bias trained; loss = binary cross-entropy on the 5 scene
  sigmoids (1 for the picture's class, 0 otherwise; all 0 for none) plus on the highest "not a scene"
  sigmoid (1 for none, 0 otherwise); AdamW, learning rate 1e-5, 5 epochs, batch 16, seed 0.
  Output per picture: the same 6 numbers as now (5 scene sigmoids, highest "not a scene" sigmoid).
- LLaVA-OneVision 7B: LoRA (rank 16, alpha 32, dropout 0.05) on the language model's attention
  projections; vision tower frozen; one training example per picture and scene question, with the
  question wording of the thesis ("Does this picture show a family scene or a family portrait? Answer
  yes or no."), target "Yes" or "No" from the label; loss on the answer token only; AdamW, learning rate
  1e-4, 1 epoch, effective batch 8, pictures at most 640 px on the long side as in the thesis.
  Output per picture: the same 13 yes-strengths, P(yes)/(P(yes)+P(no)) from the first token, read from
  the full next-token distribution.
- Zero-shot reference for LLaVA: the base model scored with the same code (so that only the training
  differs). For SigLIP the stored zero-shot scores are the reference (same preprocessing).

## Measures (over the 200 redraws, paired with zero-shot)
Primary: balanced accuracy of the scene class with the system's combination (the joint mapping on all
27 numbers), fine-tuned minus zero-shot.
Secondary: the same for the scene matcher alone and the describer alone; approval at alpha = 0.05 for
the scene class; scene presence on the 1,250 pictures (ranking AUROC and approval at alpha = 0.05 with
the corrected design).
Decision rule as in the thesis: a difference is established only if the whole 2.5-97.5% range over the
redraws lies on one side of zero. All measures are reported whatever their direction.

## Not covered
The random-archive tests of Chapter 7 (they would need a model trained on all folds and new scoring of
the archive pictures); the object classes; any change of the cut-off procedure.
