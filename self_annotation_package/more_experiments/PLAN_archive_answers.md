# The corrected design's class answers across the whole archive (plan, 9 Nov 2026, before running)

The class tasks were measured only on labelled pictures. This applies the system, as described in
Chapter 6, to all 12,110 pictures, to show what it answers for the archive. It measures no accuracy
(the archive scene-class test, archive_classes/PLAN.md, does that) and reports no approvals, because a
cut-off calibrated on the pool is not valid for the archive (Section 6.14).

- Scene class: the joint mapping fitted on all 992 class-labelled pool pictures (one rule, planned fit,
  C = 1); for each picture the most probable of the six answers and its probability.
- Objects: the per-class mappings of Task O fitted on all 300 annotated pictures, the three agents'
  probabilities averaged; a class counts as present at probability >= 0.5.
- Scene at all: the three presence mappings fitted on the 1,250-picture pool, 0.6/0.2/0.2, line 0.5.
Reported for the 10,860 pictures outside the pool (the pool pictures are in-sample): the share given
each scene answer; the mean top probability; the share with top probability below 0.5 (unsure); for
objects the share of pictures with each class and classes per picture, beside the annotated 300's
shares; agreement between "scene at all" and the scene-class answer (scene vs no scene).
Also written: a per-picture CSV (picture, scene answer and probability, object probabilities) as the
catalogue a person would start from.
