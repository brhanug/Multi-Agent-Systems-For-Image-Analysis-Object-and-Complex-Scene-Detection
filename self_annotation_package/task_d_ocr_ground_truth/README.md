# Task D — Real OCR ground truth (fixes Slide 3.9: Agent 5 has no formal precision/recall)

## Why this exists
Agent 5 (Grounded OCR / Kosmos-2.5) is corpus-wide classified into 3 buckets by a *mechanical* heuristic
(word-repetition ratio + caption-phrase pattern matching), not real ground truth: 49.8% degenerate
repetition, 3.8%+ hallucinated captioning, 46.4% "candidate real OCR" (unverified). Only 3 images were
ever visually spot-checked. This task builds real transcriptions to check that heuristic against.

## What this task does
`images/` contains **53 images**: 40 sampled from the unverified "candidate real OCR" bucket (the ones
that most need checking), plus 8 from "degenerate repetition" and 5 from "hallucinated captioning" as
calibration controls (to confirm the mechanical classifier is actually right on the buckets it claims to
already know). `labeling_sheet.csv` has one row per image, with Kosmos-2.5's actual output shown for
reference (`kosmos_output`) — don't let it bias you, read the image first.

## What to do
For each row: open `images/<image_file>`, read whatever text is actually printed/inscribed on it, and
type it into `ground_truth_transcription` as best you can (German Fraktur/blackletter is genuinely hard —
partial or approximate transcription is fine, better than nothing). If there's no legible text at all,
write `NO_TEXT`.

## When you're done
Send me the filled-in CSV. I'll compute real word/character error rate for Kosmos-2.5 against your
transcriptions, and a real precision figure for the mechanical bucket classifier (does "candidate real
OCR" actually mean real OCR, and does "degenerate repetition" actually mean no usable text) — replacing
the current 3-example spot-check with an actual number for Section 6.3 / Slide 3.9.
