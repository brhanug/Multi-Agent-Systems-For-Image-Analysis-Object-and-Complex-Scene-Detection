# Other things to do (not fixable by self-labeling)

## 1. Finish the Europeana pilot (n=24) — already set up, just needs doing
This is separate from the four tasks in this folder because the infrastructure already exists:
`scripts/core_pipeline/setup_europeana_labelstudio.py` already loaded all 24 real Europeana images into
a Label Studio project ("Europeana REAL Pilot (genuine API images)").
- Start Label Studio (however you normally do — `label-studio start`; it wasn't running when I checked).
- Open that project, label all 24 images (same has_scene schema as everything else).
- Run `scripts/core_pipeline/extract_europeana_annotations.py` to pull the results out of Label Studio's
  database.
- This closes the last open piece of the Europeana pilot: right now the thesis can only say the
  uncertainty signal *responds* to domain shift (mean fusion score 0.332 vs. 0.764) — with real labels,
  it can say whether error-detection actually *transfers* to a new archive without recalibration.

## 2. The deployed Tier 3 pair (Gemini 2.5 Flash / Claude 3.5 Sonnet) is still untested
Everything done this session (the local Qwen-VL/InternVL/LLaVA cross-model experiments) is evidence for
the *architecture's rationale*, not a test of the actual models Tier 3 deploys. To close this for real:
- Add credit to an Anthropic API key (the one pasted earlier this session had none, and is now exposed in
  plaintext in this conversation history — rotate it regardless of whether you fund it).
- Get a working Gemini API key (not yet provided at all).
- Re-run `compare_coordinators.py` / `dual_coordinator_orchestrator.py` for a real measurement.

## 3. If Ollama fixes its Llama-3.2-Vision bug, redo the cleanest version of the cross-model test
The originally-intended pairing (Llama-3.2-Vision vs. a non-Qwen-lineage model) couldn't run because of a
confirmed, currently-open upstream Ollama bug (`unknown model architecture: 'mllama'`, GitHub issues
#16490/#16547). Worth a 2-minute check every so often (`ollama pull llama3.2-vision:90b && ollama run
llama3.2-vision:90b` against any image) — if it starts working, rerunning
`scripts/core_pipeline/run_cross_model_dual_critic_ablation.py` with `MODEL_A = "llama3.2-vision:90b"`
gives a fully lineage-disjoint result instead of the Qwen2.5-VL-based proxy.

## 4. Once any Task A–D comes back, I turn it into a real number
Not something you need to do — just flagging that sending me a finished CSV isn't the end of the loop for
that item, it's the start of the next step (rebuild the corrected pool / compute precision-recall /
update the thesis and slides). Send them whenever, even one at a time.

## 5. Before the actual defense
- **Read both PDFs start to finish yourself, once, now.** A huge amount changed this session (the
  negative-control correction, the CRC reversal, the cross-model experiment, a full slide reorder) —
  worth being the one to catch anything that reads oddly before an examiner does.
- **Consider a heads-up to Prof. Mandl / Shereen Elsayed** before the defense if they saw an earlier
  draft: the CRC finding specifically *reversed* (was reported as failing, now confirmed to hold) since
  whatever version they last reviewed. Better they hear the "we found a bug and fixed it" story from you
  first than notice a changed number during questions.
- **Rehearse against the new slide order** (3.5→3.6→3.7→3.8→3.9 changed meaning this session) so the
  verbal walkthrough matches what's now on screen.
