# E6. Fixed or learned combination weights for the class tasks (plan fixed 8 Nov 2026, before running)

Question: the combination weights 0.6 / 0.2 / 0.2 were inherited from the deployed design. For scene
presence, equal and learned weights were already shown to give the same results. Is a learned
combination better for the scene classes and the object classes?

Variants, all fitted on training books only (nothing on calibration or test books):
- fixed: 0.6 / 0.2 / 0.2 (scene classes) or the mean of the three (objects), as reported;
- equal weights (scene classes);
- learned weights (stacking): each agent's mapping is fitted on four fifths of the training books and
  predicts the remaining fifth (5 folds by book); a logistic regression on these out-of-fold
  probabilities (scene classes: one multinomial regression on the 3 x 6 probabilities; objects: one
  regression per class on the 3 probabilities) learns how to combine the agents; the agents' mappings
  are then refitted on all training books;
- one joint mapping: a single regression on all agents' raw reports together (scene classes: 27
  numbers; objects: per class, the 3 scores);
- the describer alone.

Protocol and measures as in Tasks S and O (class_tasks/PLAN.md): scene classes 200 book-disjoint
redraws 60/20/20 on 992 pictures, planned (unweighted) mappings, measures macro-F1, balanced accuracy,
share approved at alpha 0.05 with the share of approved answers wrong; objects 200 draws 40/30/30 on
the 300 pictures, measures mean AUROC, mean AP, and at alpha 0.10 the missed-object list (miss rate,
classes listed, wrong classes per picture). Differences against the fixed combination are paired per
redraw.

Reading: a learned combination is better if its 2.5-97.5 range against the fixed one excludes zero
(with the caveat that the redraws share pictures).
