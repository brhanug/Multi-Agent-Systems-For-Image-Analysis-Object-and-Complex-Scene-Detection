# Plan: precision of Agent 4's archive-wide links (Section 8.2)

Written 21 Nov 2026, before any judgement is made.

## Why
Across all 12,110 pictures, 2,434 have a nearest neighbour in another book at SigLIP similarity
>= 0.92. The 100% precision at that level was measured on the top pairs of a 1,500-picture sample;
the archive-wide links are unmeasured (Table 9.1).

## Items
The nearest-neighbour links recomputed with scripts/signal_rebuild/run_agent4_embeddings_full_corpus.py
(link function, same embeddings file); the rebuild must give 2,434 links at >= 0.92 or the run
stops. 50 links drawn uniformly at random with seed 20261121; a pair drawn twice (A->B and B->A) is
judged once and replaced by the next draw, so 50 distinct pairs are judged.

## Judgement (blind to similarity)
The two pictures side by side: same picture (a reprint or the same scan) / same style only /
unrelated / unsure. The similarity value is not shown.

## Measures, fixed now
- Precision: share "same picture" among the judged pairs (unsure left out and counted), with a
  Wilson 95% interval.
- The same split by whether the two pictures have the same page number (same book digitised twice)
  or not.
The result is added to Section 8.2 and Table 9.1, whatever it is.
