# Task A — Confirm or correct the "negative control" pool (fixes the Slide 4.2 blind spot)

## Why this exists
Your thesis's negative-control pool (`human_spatial_audit/user_annotations_800.csv`, rows 801–1999)
turned out to be 1,199 catalog records defaulted to "no scene" that were never actually reviewed by a
human. A 144-image spot-check (already done, last session) found 66.7% of them were wrong — real scenes
mislabeled as empty. That correction is already in the thesis, but it left only **7 confirmed negative
images** in the test split — too few to say with confidence whether the "disagreement is blind to
negative controls" finding (Slide 4.2, Slide 3.7) is real or just noise.

## What this task does
`images/` contains **360 images** sampled from the 1,055 images in that pool I have *not* yet checked
(no overlap with the 144 already done). `labeling_sheet.csv` has one row per image.

## What to do
For each row, open the image (`images/<image_file>`) and fill in `your_label`:
- `scene` — the image depicts a person, animal, landscape, or any describable narrative content.
- `no_scene` — pure text, sheet music, blank space, or non-representational decorative ornament only.

Use `notes` for anything ambiguous — don't agonize over borderline cases, just flag them.

## Why 360
At the ~33% true-negative rate found in the first 144-image sample, ~360 more images should yield
roughly 100+ more confirmed negatives — enough to take the test-split negative count from 7 to somewhere
in the 15–30 range after group-level splitting, a real improvement in statistical power.

## When you're done
Send me the filled-in CSV. I'll rebuild the corrected pool (`corrected_clean_pool.csv`), rebuild the
group-level split, and re-run the CRC/AWLF/SAA-decomposition scripts — all already built and working
(`build_corrected_negative_control_pool.py`, `build_corrected_clean_holdout_split.py`,
`run_corrected_clean_holdout_evaluation.py`, `run_corrected_clean_holdout_crc.py`) — then update the
thesis/slides with the new, better-powered numbers.

## Bonus: this can also fix the inter-annotator kappa gap
If you have time, do this task **blind** (don't look at `current_label` while labeling) — your labels
can then be compared against my prior single-rater pass on the *same 144 images I already did* (ask me
for that image set too) to compute a real, purpose-built Cohen's kappa, fixing the other open item
(thesis Limitations: "Larger-scale inter-annotator study").
