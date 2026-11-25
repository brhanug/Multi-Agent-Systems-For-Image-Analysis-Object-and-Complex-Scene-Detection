#!/usr/bin/env python3
"""
check_agent5_ocr_degeneracy.py
---------------------------------
A real, corpus-wide benchmark of Agent 5 (Kosmos-2.5 grounded OCR), run to
close a previously-documented gap: Agent 5 was implemented and deployed
corpus-wide, but never independently checked against what its outputs
actually contain.

Classifies every one of the 12,110 deployed results/kosmos_grounding.jsonl
outputs into three mechanically-detectable buckets, each validated against
the source images by direct visual inspection (see the thesis, Agent 5
section, for the specific examples):

  1. Degenerate repetition: autoregressive text models can fail by looping
     (repeating a token or short phrase until the length limit). Detected
     via the ratio of unique words to total words -- a low ratio is
     consistent with looping rather than genuine transcription. Visually
     confirmed: a magic-lantern slide with clearly legible printed text
     ("Hartmann's Chocolade") whose actual output is just "the" repeated.

  2. Hallucinated captioning: fluent, grammatical prose describing generic
     image content ("a picture of a man with a gun") instead of reading
     any text -- detected via a regex over common image-caption openers.
     This is a LOWER BOUND (other phrasings exist uncaught). Visually
     confirmed: two birds in a nest, captioned "a picture of a man with a
     gun" / "a picture of a woman with a gun" -- completely unrelated to
     the actual image.

  3. Candidate real OCR (the remainder): not mechanically distinguishable
     from genuine-but-degraded transcription without visual inspection.
     Visually confirmed on a sample: a German Fraktur table-of-contents
     page transcribed with all five page numbers exactly correct and
     several words closely recognizable, despite heavy character-level
     noise -- a genuine transcription attempt, not hallucination.

Outputs:
  results/multi_agent/agent5_ocr_degeneracy_check.json
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
KOSMOS_JSONL = BASE / "results" / "kosmos_grounding.jsonl"
OUT_JSON = BASE / "results" / "multi_agent" / "agent5_ocr_degeneracy_check.json"

LOW_RATIO_THRESHOLD = 0.3
CAPTION_RE = re.compile(
    r'!\[(the image (above|below)?|a picture of|the picture|an image of)',
    re.IGNORECASE,
)


def unique_word_ratio(text: str) -> float:
    words = text.split()
    if len(words) < 5:
        return 1.0  # too short to meaningfully judge as a repetition loop
    return len(set(words)) / len(words)


def main() -> None:
    rows = [json.loads(line) for line in KOSMOS_JSONL.open()]
    n = len(rows)

    n_degenerate = n_caption = n_other = 0
    ratios, lengths = [], []
    for r in rows:
        text = r["kosmos_output"]
        ratio = unique_word_ratio(text)
        ratios.append(ratio)
        lengths.append(len(text))
        if ratio < LOW_RATIO_THRESHOLD:
            n_degenerate += 1
        elif CAPTION_RE.search(text[:120]):
            n_caption += 1
        else:
            n_other += 1

    result = {
        "n_total": n,
        "bucket_1_degenerate_repetition": n_degenerate,
        "frac_degenerate_repetition": round(n_degenerate / n, 4),
        "bucket_2_hallucinated_captioning_min": n_caption,
        "frac_hallucinated_captioning_min": round(n_caption / n, 4),
        "bucket_3_candidate_real_ocr": n_other,
        "frac_candidate_real_ocr": round(n_other / n, 4),
        "mean_unique_ratio": round(statistics.mean(ratios), 4),
        "median_unique_ratio": round(statistics.median(ratios), 4),
        "mean_char_length": round(statistics.mean(lengths), 1),
        "median_char_length": statistics.median(lengths),
        "note": "Bucket 2 is a LOWER BOUND (regex catches only common caption "
                "openers). Bucket 3 is a remainder, not a positive confirmation "
                "of correctness -- it contains a mix of genuine-but-degraded "
                "OCR (visually confirmed on a sample) and possibly some "
                "uncaught bucket-1/2 cases. All three buckets are visually "
                "validated with real examples in the thesis, not asserted "
                "from the heuristic alone.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"n = {n:,}")
    print(f"Bucket 1 (degenerate repetition):        {n_degenerate:,} ({n_degenerate/n:.1%})")
    print(f"Bucket 2 (hallucinated captioning, min):  {n_caption:,} ({n_caption/n:.1%})")
    print(f"Bucket 3 (candidate real OCR, remainder): {n_other:,} ({n_other/n:.1%})")
    print(f"Mean unique-word ratio: {result['mean_unique_ratio']}")
    print(f"\n✅ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
