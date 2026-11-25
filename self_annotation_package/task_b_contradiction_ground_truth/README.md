# Task B — Gold contradiction labels (fixes Slide 3.9: Agent 2 has no formal precision/recall)

## Why this exists
Agent 2 (Contradiction Detector) flags a candidate hallucination whenever the VLM mentions something
YOLO didn't detect. The thesis is honest that this has never been scored against real gold labels —
only 15 candidates were ever spot-checked (3 in detail). This task builds the real gold set.

## What this task does
`images/` contains the **25 real images** where the pipeline's own VLM scene-graph output and YOLO
detections actually disagree (pulled directly from `final_dataset/metadata/scene_graphs_v1.jsonl` +
`results/multi_agent/upgraded_agent0_fusion.json` — the exact data Agent 2 itself runs on, not a new
sample). `labeling_sheet.csv` has **127 rows**, one per (image, claimed-but-undetected noun) pair —
several rows per image is normal, since one image can have several disputed nouns.

## What to do
For each row: open `images/<image_file>`, look for `claimed_noun` (the thing the VLM said was there
that YOLO didn't detect — `yolo_detected_instead` shows what YOLO found instead, for context). Fill in
`your_judgment`:
- `present` — the claimed noun really is visible in the image (a genuine YOLO miss, not a hallucination).
- `absent` — the claimed noun is not in the image (a genuine VLM hallucination).
- `ambiguous` — genuinely can't tell (small, occluded, stylized).

This is exactly the judgment the thesis's own spot-check already made on 3 cases — you're just extending
it to the full 127.

## When you're done
Send me the filled-in CSV. I'll compute real precision/recall for Agent 2 (fraction of VLM's own flagged
contradictions that are `absent` = true hallucinations, vs. `present` = false alarms from YOLO
under-detection) and update Section 6.1 / Slide 3.9 with the real number in place of "not independently
benchmarked."

## Bonus: reuse for the Tier 3 cross-model critique verification
The same `present`/`absent`/`ambiguous` judgment is what's needed to turn the Tier 3 local disjoint-model
experiment's raw flag rates (54% and 18%, thesis §2, Slide A.5) into real precision numbers instead of
"visually spot-checked 5 cases." If you have appetite for more after this, ask me for that CSV
(`results/multi_agent/cross_model_dual_critic_ablation*.csv`) — same task, different image set.
