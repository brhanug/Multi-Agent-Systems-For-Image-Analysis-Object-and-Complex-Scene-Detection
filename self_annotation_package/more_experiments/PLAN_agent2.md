# Agent 2 as it runs now, scored on the 300 hand-annotated pictures (plan, 9 Nov 2026, before running)

Agent 2's gold precision (39.7%, 127 candidates) comes from an early prototype: 25 pictures of a 50-picture
scene-graph file, with an earlier detector. The Agent 2 that runs on the archive now
(run_tier2_tier3_from_full_tier1.py: the VLM's free-text answers mapped to the detector's taxonomy by a
word list; a flag for every class the VLM names that the detector, at confidence 0.25, did not return)
was never scored. The 300 pictures annotated with the eight object classes allow this without new labels.

Pictures: the 300 of human_class_truth_300.csv. Classes: the eight shared ones (person, child, animal,
building, weapon, vehicle, tree, text); the detector's "horse" counts as animal, "clothing" is outside the
eight and is left out. A flag on class c is a hallucination if the annotation says c is absent, a detector
miss if present.

Reported, for the deployed detector (yolo_full_corpus.jsonl) and, as the corrected design would run it,
the retrained object finder (new_student_class_scores.jsonl, count at 0.25):
- flags, pictures with a flag, precision as a hallucination detector (hallucinations / flags) with a
  Wilson 95% interval, per class;
- recall: of all VLM hallucinations among the eight classes (VLM names c, annotation says absent), the
  share Agent 2 flags (it misses those the detector also wrongly reports);
- the share of VLM claims that are hallucinations at all (the base rate a flag must beat).
Reading: Agent 2 works as a hallucination detector if its precision clearly exceeds the base rate.
