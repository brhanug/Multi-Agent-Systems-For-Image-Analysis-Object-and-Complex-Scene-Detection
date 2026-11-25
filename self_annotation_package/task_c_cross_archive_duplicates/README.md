# Task C — Real cross-archive duplicate/reprint labels (fixes Slide 3.9: Agent 4 has no formal benchmark)

## Why this exists
Agent 4 (Cross-Archive Linker) finds recurring compositions across different publication series using
SigLIP visual embeddings. The thesis only ever verified 2 pairs by hand (one genuine match at sim=0.938,
one style-only false match at sim=0.901). This task builds a real, larger gold set.

## What this task does
Real SigLIP embeddings were computed over 1,500 real corpus images (extending the same method already
in `benchmark_agent4_visual_embedding_linking.py`), and the **80 highest-similarity cross-archive pairs**
(similarity 0.55–0.985, all genuinely computed, not hand-picked) were kept. `images/` has both images
of each pair (`NNN_A_...jpg` / `NNN_B_...jpg`); `labeling_sheet.csv` has one row per pair.

## What to do
For each row: open both `image_a` and `image_b` side by side. Fill in `your_judgment`:
- `same_composition` — genuinely the same illustration/photo (a real reprint or duplicate scan).
- `same_style_only` — different picture, but shares engraving style/genre/period costume (a
  false-positive-for-recurrence, like the sim=0.901 case already documented).
- `unrelated` — no meaningful visual relationship beyond the embedding's own similarity score.

## When you're done
Send me the filled-in CSV. I'll compute a real precision-at-similarity-threshold curve for Agent 4
(what fraction of pairs above each similarity cutoff are genuine `same_composition` matches) — replacing
the current "2 hand-checked examples" with an actual benchmark, and giving a real, evidence-based
similarity threshold recommendation for the deployed system instead of a guess.
