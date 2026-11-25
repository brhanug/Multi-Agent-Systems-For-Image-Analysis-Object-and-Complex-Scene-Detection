# The five scene classes and the eight object classes as the thesis tasks (plan fixed 8 Nov 2026)

Written and committed before either analysis below was run. The labels already exist; what is fixed
here is the method, the measures and how they are read, so that nothing is chosen after seeing results.
Anything added later is reported as an extra analysis.

## Why
The class lists were frozen earlier: five scene classes (teaching, family, playing, landscape, drawing)
and eight object classes (person, child, animal, building, weapon, vehicle, tree, text). The thesis so
far evaluated scene presence (yes / no) as its main decision. These two analyses make the frozen
classes the tasks: which scene class a picture shows, and which object classes it contains.

## Task S: scene class (six answers)
Answers: the five scene classes and "no scene".

Pictures: the labelled pool (results/multi_agent/corrected_clean_holdout_split.csv) wherever the class
is known:
- the 801 expert-reviewed pictures, with the expert's scene class (each has at most one), or "no scene"
  if none was given;
- the 191 pool pictures labelled "no scene" in the 2026 rounds.
The 258 pool pictures labelled "scene" in the 2026 rounds carry no class and are left out (counted).

Labels, two variants:
- one rule (primary): as above, but an expert picture the author labelled "no scene" under the scene
  rule (self_annotation_package/scene_consistency, final_check) becomes "no scene"; "scene" and
  "unsure" keep the expert class. Counts: no scene 486, drawing 341, landscape 76, family 51,
  playing 32, teaching 6 (992 pictures).
- as labelled: the expert classes as given (drawing 624, no scene 196, ...).

Agents and their reports (the same agents as the corrected design):
- object finder (retrained student): 8 best-box scores;
- scene matcher (SigLIP): 5 scene sigmoids and the "no scene" score;
- describer (LLaVA-OneVision): 5 scene-class yes-strengths and 8 object-class yes-strengths.

Method: 200 book-disjoint redraws (60/20/20, seed 20260926). On the training books each agent's
report is mapped to a probability over the six answers by a multinomial logistic regression (C = 1);
the three are combined with the weights 0.6 / 0.2 / 0.2; the answer is the most probable class; the
uncertainty is 1 - (largest probability).
- Risk control (top-1): conformal risk control on the calibration books, loss = approved and wrong
  class, alpha 0.05 and 0.10.
- Prediction sets: the smallest set {k : p_k >= 1 - lambda} (always including the top class) whose
  miss rate on the calibration books keeps the conformal bound at alpha 0.05 and 0.10.

Measures (test books, mean and 2.5/97.5 percentiles over redraws):
- primary: macro-F1 over the six answers, and the share approved at alpha 0.05 with its realised risk;
- accuracy, balanced accuracy, per-class precision / recall / F1, the confusion matrix pooled over
  redraws;
- prediction sets: coverage, mean size, share of one-class sets, coverage per class.

Compared, paired per redraw: each agent alone; the three with equal weights; the scene matcher
zero-shot (no training: the largest of its six scores); the majority answer.

Reading: teaching has 6 pictures and is reported but not interpreted. A difference between the
combination and an agent alone counts as established only if its 2.5-97.5 range over redraws excludes
zero, and the thesis states that this range understates the uncertainty (the redraws share pictures).

## Task O: object classes (eight, several per picture)
Pictures: the 300 hand-annotated pictures (results/signal_rebuild/class_scores/human_class_truth_300.csv),
a class present if a human box of that class exists.

Systems: object finder (retrained student, best-box score), scene matcher (SigLIP class score),
describer (LLaVA yes-strength), and their combination (mean of the three after a logistic mapping per
system and class).

Method: 200 redraws, books split 40 / 30 / 30 (fit / calibrate / test, seed 20260920). Mappings are
fitted on the fit books; the cut-off on the calibration books; everything measured on the test books.
- Per class: AUROC and average precision of presence.
- A promise on missed objects: each picture gets the set of classes whose probability is at least t;
  conformal risk control chooses t so that the expected share of a picture's present classes left out
  of its set (false negative rate) is at most alpha, alpha 0.05, 0.10, 0.20.
  Measured: realised miss rate, wrongly added classes per picture, precision of the set, recall per
  class, share of pictures whose set is exactly right.

Box level, quoted from the existing result (results/signal_rebuild/student_rebuild/
teacher_candidates_300.json): AP50 per class of the original and the retrained object finder.

## Not changed
The archive tests (final_check, confirm_check) labelled scene presence only; they stay as tests of the
first level of the scene decision. A test of the scene classes on random archive pictures would need
new labels and its own plan.
