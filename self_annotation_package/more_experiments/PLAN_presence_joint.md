# A learned combination for scene presence too? (plan, 9 Nov 2026, before running)

The corrected design combines the three agents' presence probabilities with the hand-set 0.6/0.2/0.2
(scene at all), while the scene classes use one joint mapping on all agent reports. Equal and stacked
weights were already compared (fusion_weights_corrected.json) and gave the same results. Not yet tested:
one joint logistic regression on all 22 numbers the agents report for presence (8 object scores, 6 scene
scores, 8 describer yes-strengths), the form the scene classes use.

Same 1,250 pool pictures, labels as used for the main results, the same 200 book-disjoint 60/20/20
redraws (seed 20260926), C = 1 as for every other mapping. Measured: scene AUROC, test error, approval
and realised risk at alpha 0.05 and 0.10 under confidence triage (u = 1 - |p - 0.5|); paired differences
against the fixed weights, read by the thesis rule (range excludes zero) and the corrected resampled
t-test. Reading: if no difference is established, the fixed weights stay for scene presence (the archive
tests were fixed with them) and the thesis states that a learned combination gives the same.
