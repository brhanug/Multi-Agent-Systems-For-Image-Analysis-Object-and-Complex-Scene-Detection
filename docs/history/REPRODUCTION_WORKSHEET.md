# Reproduction Worksheet

**Thesis:** *Knowing When to Trust AI on Historical Images* — Brhanu Atsbaha, Universität Hildesheim
**Purpose:** reproduce every result in the thesis from the terminal, start to end, and check each one.

Work through it top to bottom. Tick each box as you go. Every command is meant to be copied exactly.

---

## 0. Read this first

### Two modes — know which one you are running

| Mode | Command form | What it does | Safe? |
|---|---|---|---|
| **VERIFY** | `python script.py --verify` | Recomputes the result and compares it to the committed file. Prints `ALL FIGURES REPRODUCE` or `MISMATCH`. **Writes nothing.** | ✅ Always safe |
| **REGENERATE** | `python script.py` | Recomputes and **overwrites** the committed result file. | ⚠️ Changes committed files |

**Use VERIFY for this whole worksheet.** It is the reproduction check. Only use REGENERATE if you have deliberately changed an input and want new results — and then commit the change.

The one exception is **Stage 1** (Tier 1 inference), which has no verify mode because it *is* the raw input. It is resumable and writes only to `results/signal_rebuild/`.

### Time budget

| What | Time |
|---|---|
| Stage 0 — environment check | 1 min |
| Stage 1 — Tier 1 inference, all 4 models, all 12,110 images | **~4 h 40 min** (optional — outputs already exist) |
| Stages 2–9 — every result, verified | **~10 min** |
| Stage 10 — compile all documents | ~3 min |

**If you only want to confirm the numbers, skip Stage 1** — its outputs are already on disk, and Stages 2–9 check everything downstream of them in about ten minutes.

### Notation used below

```bash
cd /data/brhanu/thesis_project
PY=.venv/bin/python3
```

Run those two lines once at the start of every terminal session. Every command below assumes them.

---

## Stage 0 — Environment check  *(1 min)*

- [ ] **0.1** Enter the project and set the interpreter

```bash
cd /data/brhanu/thesis_project
PY=.venv/bin/python3
$PY --version                      # expect: Python 3.12.3
```

- [ ] **0.2** Confirm the libraries and versions

```bash
$PY -c "
import torch, transformers, vllm, ultralytics, pandas, numpy, sklearn, scipy, PIL
print('torch       ', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('vllm        ', vllm.__version__)
print('ultralytics ', ultralytics.__version__)
print('pandas      ', pandas.__version__)
print('numpy       ', numpy.__version__)
print('sklearn     ', sklearn.__version__)
print('scipy       ', scipy.__version__)
"
```

Expected: torch `2.9.0+cu128` (cuda True, 4 GPUs), transformers `4.57.3`, vllm `0.13.0`, ultralytics `8.3.241`, pandas `2.3.3`, numpy `2.2.6`, sklearn `1.9.0`, scipy `1.16.3`.

- [ ] **0.3** Confirm the input data is present

```bash
for p in final_dataset/metadata/manifest.csv \
         final_dataset/images/restored \
         human_spatial_audit/user_annotations_800.csv \
         final_dataset_v1_refresh/metadata/vlm_outputs/groundingdino \
         runs/detect/yolo11_v2_augmented_refresh/weights/best.pt \
         results/multi_agent/corrected_clean_holdout_split.csv; do
  [ -e "$p" ] && echo "present  $p" || echo "MISSING  $p"
done
```

Expected: six `present` lines. Any `MISSING` must be fixed before continuing.

- [ ] **0.4** Confirm the corpus size

```bash
find final_dataset/images/restored -name '*.jpg' | wc -l      # expect: 12110
```

- [ ] **0.5** Confirm the GPUs are free

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

Expected: four RTX A6000, near-zero memory used. Needed only for Stage 1.

---

## Stage 1 — Tier 1 inference: all four models on all 12,110 images  *(~4 h 40 min, OPTIONAL)*

> **Skip this stage if you only want to verify the numbers.** Its outputs already exist.
> Run it only to regenerate the raw model outputs from the images themselves.

This runs YOLOv11, LLaVA-OneVision, Kosmos-2.5 and SigLIP over every image. Measured throughput on four A6000s:

| Model | Script | GPUs | Throughput | Time |
|---|---|---|---|---|
| LLaVA-OneVision (VQA, 4 questions/image) | `run_llava_full_corpus.py` | 4 shards | 2.56 img/s | 80 min |
| Kosmos-2.5 (grounded OCR) | `run_kosmos_full_corpus.py` | 4 shards | 1.24 img/s | 2 h 47 min |
| SigLIP (scene) | `run_siglip_scene_full_corpus.py` | 1 | 6.7 img/s | 30 min |
| YOLOv11 (detection) | `run_yolo_full_corpus.py` | 1 | 16.2 img/s | 12 min |

### Option A — run the whole stage with one command (recommended)

- [ ] **1.1** Launch it detached, so it survives closing the terminal

```bash
setsid nohup bash scripts/signal_rebuild/run_full_tier1_pipeline.sh < /dev/null > /dev/null 2>&1 &
disown
```

- [ ] **1.2** Watch progress

```bash
tail -f results/signal_rebuild/full_tier1_run.log
```

Press `Ctrl+C` to stop watching — the run keeps going. It is finished when the log prints:

```
=== ALL STAGES COMPLETE ===
```

- [ ] **1.3** Check live progress of each model at any time

```bash
echo "LLaVA : $(cat results/signal_rebuild/llava_full/shard_*.jsonl 2>/dev/null | wc -l) / 12110"
echo "Kosmos: $(cat results/signal_rebuild/kosmos_full/shard_*.jsonl 2>/dev/null | wc -l) / 12110"
echo "SigLIP: $(wc -l < results/signal_rebuild/siglip_scene_full.jsonl 2>/dev/null || echo 0) / 12110"
echo "YOLO  : $(wc -l < results/signal_rebuild/yolo_full_corpus.jsonl 2>/dev/null || echo 0) / 12110"
```

- [ ] **1.4** If it stops for any reason, **just run 1.1 again.** Every model skips images it has already written, so it continues rather than restarts.

### Option B — run each model by hand

```bash
# YOLOv11 detector, one GPU (~12 min)
$PY scripts/signal_rebuild/run_yolo_full_corpus.py --device 0 --batch 48

# LLaVA-OneVision, four shards in parallel (~80 min)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/signal_rebuild/run_llava_full_corpus.py \
    --shard $i --num-shards 4 > results/signal_rebuild/llava_shard_$i.log 2>&1 &
done; wait

# Kosmos-2.5, four shards in parallel (~2 h 47 min)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/signal_rebuild/run_kosmos_full_corpus.py \
    --shard $i --num-shards 4 > results/signal_rebuild/kosmos_shard_$i.log 2>&1 &
done; wait

# SigLIP scene classifier, one GPU (~30 min)
CUDA_VISIBLE_DEVICES=0 $PY scripts/signal_rebuild/run_siglip_scene_full_corpus.py
```

### To regenerate from scratch instead of resuming

The scripts resume by skipping images already written. To force a genuinely fresh run, move the old outputs aside first — **do not delete them**, so you can compare:

```bash
mkdir -p results/signal_rebuild/_previous_run
mv results/signal_rebuild/llava_full            results/signal_rebuild/_previous_run/ 2>/dev/null
mv results/signal_rebuild/kosmos_full           results/signal_rebuild/_previous_run/ 2>/dev/null
mv results/signal_rebuild/siglip_scene_full.jsonl results/signal_rebuild/_previous_run/ 2>/dev/null
mv results/signal_rebuild/yolo_full_corpus.jsonl  results/signal_rebuild/_previous_run/ 2>/dev/null
```

- [ ] **1.5** Confirm all four are complete

```bash
echo "YOLO  : $(wc -l < results/signal_rebuild/yolo_full_corpus.jsonl)"
echo "LLaVA : $(cat results/signal_rebuild/llava_full/shard_*.jsonl | wc -l)"
echo "Kosmos: $(cat results/signal_rebuild/kosmos_full/shard_*.jsonl | wc -l)"
echo "SigLIP: $(wc -l < results/signal_rebuild/siglip_scene_full.jsonl)"
```

Expected: `12110` for all four.

---

## Stage 2 — The signal table everything rests on  *(~5 s)*

Every result in the thesis reads `results/multi_agent/agent_comparison_scores.csv`. This stage proves it regenerates from source — and settles *how*.

- [ ] **2.1** Re-execute the fusion stage over all 12,110 images, two ways

```bash
$PY scripts/signal_rebuild/verify_pipeline_reproduces_corpus.py --verify
```

Expected: `ALL FIGURES REPRODUCE`

**What it reproduces:** the fusion stage, run twice into a temporary directory. The live-tree route (detector files absent → manifest fallback) regenerates the committed table **bit-for-bit**, all 5 columns, maximum difference 0. The archived-files route does not (`A_object` 11.6%, `A_scene` 0%), because those real model outputs cover only **231** of 12,110 images. This is the execution-level proof that `A_object` comes from the fallback. *(Thesis §2.4)*

- [ ] **2.2** Confirm the decision rule is safe and equivalent

```bash
$PY scripts/core_pipeline/saa_decision.py
```

Expected: `RESULT: SAFE AND EQUIVALENT` — 0 disagreements across 12,110 images, closest attainable score 0.05 from the threshold. *(Thesis §4.2.1, the knife-edge fix)*

---

## Stage 3 — The evaluation pools  *(~1 s)*

- [ ] **3.1** Rebuild the corrected negative-control pool from the human annotations

```bash
$PY scripts/core_pipeline/build_corrected_negative_control_pool.py --verify
```

Expected: `VERIFY vs committed corrected_clean_pool.csv: IDENTICAL`

**What it reproduces:** the 1,250-image corrected pool and its PPN-disjoint split (749 train / 252 calibration / 249 test). The original 1,199-image negative-control pool was **66.7% mislabelled**; this is the corrected version every held-out result uses. *(Thesis §4.3)*

---

## Stage 4 — The headline results  *(~5 s)*

- [ ] **4.1** Held-out evaluation: AWLF and error ranking

```bash
$PY scripts/core_pipeline/run_corrected_clean_holdout_evaluation.py --verify
```

Expected: `ALL FIGURES REPRODUCE`

**What it reproduces:** AWLF v2 held-out ROC-AUC **0.8702**; error ranking on held-out scene-positive images **0.9687** (n=211). *(Thesis §2.5, §4.3)*

- [ ] **4.2** Conformal risk control — the operational guarantee

```bash
$PY scripts/core_pipeline/run_corrected_clean_holdout_crc.py --verify
```

Expected: `ALL FIGURES REPRODUCE`

**What it reproduces:** at α = 0.05, λ̂ = **0.372**, test risk **0.0201**, bound met, **60.6%** auto-approval, **91.1%** error recall, **2.31×** lift — and the bound holding at every α from 0.01 to 0.15. *(Thesis §5.2)*

---

## Stage 5 — What the uncertainty signal actually measures  *(~6 min)*

These establish the central finding: disagreement does not beat plain confidence.

- [ ] **5.1** Decompose the uncertainty signal  *(46 s)*

```bash
$PY scripts/core_pipeline/run_uncertainty_signal_decomposition.py --verify
```

**Reproduces:** `U_SAA` in-sample **0.9724**; plain confidence **0.9717** (tied); derived term removed **0.7694**. `A_agree ≡ A_object` on 88.4% of images. *(Thesis §4.2)*

- [ ] **5.2** Can the formulation be repaired?  *(21 s)*

```bash
$PY scripts/core_pipeline/run_corrected_formulation.py --verify
```

**Reproduces:** the repair sequence **0.8600 → 0.8340 → 0.7005 → 0.6080** — each correction makes it worse. *(Thesis §4.2.3)*

- [ ] **5.3** Search nine formulations  *(75 s)*

```bash
$PY scripts/signal_rebuild/run_formulation_search.py --verify
```

**Reproduces:** nine formulations, all below plain confidence (**0.8167**); best **0.7650**, worst **0.5551**. *(Thesis §4.2.2)*

- [ ] **5.4** The ceiling — no formulation can win  *(95 s)*

```bash
$PY scripts/signal_rebuild/run_drop_agree_learn_values.py --verify
```

**Reproduces:** exhaustive evaluation of all **8! = 40,320** orderings of the 8 signal states; ceiling **0.8871**; confidence **0.8828**, gap **0.0043**. Dropping `A_agree` alone hurts (0.8600 → 0.8340); dropping and learning gives 0.8801. *(Thesis §4.2.2)*

- [ ] **5.5** Mutual-information (BALD) decomposition  *(17 s)*

```bash
$PY scripts/signal_rebuild/run_bald_decomposition.py --verify
```

**Reproduces:** BALD-style epistemic/aleatoric split, one of the nine formulations. *(Thesis §4.2.2)*

- [ ] **5.6** Non-circular retest  *(51 s)*

```bash
$PY scripts/signal_rebuild/run_noncircular_disagreement_test.py --verify
```

**Reproduces:** leave-one-model-out, where the predictor and the uncertainty signal share no input; disagreement still does not beat confidence. *(Thesis §4.2.4)*

- [ ] **5.7** Measure how circular the error label is  *(17 s)*

```bash
$PY scripts/signal_rebuild/measure_error_label_circularity.py --verify
```

**Reproduces:** the overlap between the error label and the signals that predict it. *(Thesis §4.2)*

- [ ] **5.8** Symmetric ablation of all four terms  *(67 s)*

```bash
$PY scripts/signal_rebuild/run_agree_ablation.py --verify
```

**Reproduces:** Table 4.4 — dropping `A_agree` from the spread hurts (−0.026 / −0.077), from a fitted model changes nothing (0.000 / +0.0015). `A_object` is recoverable from `(A_agree, A_vlm)` on **100%** of images. *(Thesis §4.6)*

- [ ] **5.9** Review gaps: predictive ablation, held-out calibration  *(5 s)*

```bash
$PY scripts/signal_rebuild/run_review_gaps_p2_p3.py --verify
```

**Reproduces:** removing Object costs **0.0861**, removing VLM *improves* by **+0.0108**; ECE **0.109** on the PPN-disjoint split against < 0.001 within-cohort; consensus + disagreement adds exactly 0. *(Thesis §4.5, §4.6)*

- [ ] **5.11** Redesign every signal from its model's own output  *(11 s)*

```bash
$PY scripts/signal_rebuild/run_redesigned_signals.py --verify
```

**Reproduces:** `A_object` 4 → **8,979** distinct values, `A_scene` 2 → **11,882**, `U_SAA` 8 → **1,247**. Disagreement now trails confidence by **0.278** and **0.216**, both 95% intervals entirely below zero. *(Thesis §2.4)*

- [ ] **5.12** Diagnose *why* disagreement loses  *(2 s)*

```bash
$PY scripts/signal_rebuild/diagnose_why_disagreement_fails.py --verify
```

**Reproduces:** effective rank **1.677** of 4; all three models wrong together on only **6.24%** (so heterogeneity is satisfied); within-SAA-quartile AUROC **0.34–0.42**; mean `U_SAA` differs by **0.013** between errors and correct while mean `SAA` differs by **0.290**. *(Thesis §2.4)*

- [ ] **5.13** Corpus-scale payoff, and what can be removed  *(3 s)*

```bash
$PY scripts/signal_rebuild/run_redesigned_full_corpus_and_recommend.py --verify
```

**Reproduces:** at the same α, spread triage certifies **10.0%** of the corpus for auto-approval, confidence triage **61.1%**, at higher error recall (89.6% vs 91.5%). Removing the detector collapses balanced accuracy to 0.533; removing the spread entirely costs nothing. *(Thesis §2.4)*

- [ ] **5.14** Repair the three mis-named signals, and retest  *(40 s)*

```bash
$PY scripts/signal_rebuild/run_fixed_signals.py --verify
```

**Reproduces:** deployed `A_agree` = `1 − |A_object − ½|` on **98.92%** of images (correlation only **0.18** — exact but non-linear, which is why correlation screening missed it). Repaired OCR scores the images the deployed signal calls perfect at **0.159** vs **0.378** elsewhere (r = **−0.32**). Spread over deployed signals **0.8558**, over repaired signals **0.4362** (Δ −0.413, CI [−0.513, −0.310]). Confidence alone **0.8499** → plus all three repaired signals **0.8563** (Δ +0.006, CI [−0.004, +0.018]). *(Thesis §2.4, Tables 2.5–2.6)*

- [ ] **5.15** The scene channel, and the last hypothesis  *(28 s)*

```bash
$PY scripts/signal_rebuild/run_scene_channel_fix.py --verify
```

**Reproduces:** SigLIP's softmax channel (deployed) has corpus minimum **0.2956** and cannot express absence; its sigmoid channel reaches **0.0** and is below 0.01 on **48.6%** of images. Against human scene labels the deployed channel scores **0.4172** — below chance — and the unused channel **0.7754** (+0.358, CI [+0.299, +0.415]). A fitted fusion learns coefficient **−0.162** for the deployed channel and **+0.486** for the corrected one. End to end: error rate **42.6% → 39.4%**, balanced accuracy **0.663 → 0.682**. Ensemble unanimity **100% → 14.9%** under class-balanced calibration, and disagreement then scores **0.5014** (chance) against confidence **0.8093**. *(Thesis §2.4, Tables 2.7–2.8)*

- [ ] **5.16** Apply the corrections end to end  *(6 s)*

```bash
$PY scripts/signal_rebuild/run_corrected_channel_refit.py --verify
```

**Reproduces:** refitted fusion weights **0.245 / 0.487 / 0.269** on the corrected channel (scene goes from smallest weight to largest) against **0.383 / −0.162 / 0.455** on the deployed channel — the fit prefers to *subtract* the deployed scene agent. Error rate across the four configurations: **42.2% → 37.8% → 39.4% → 36.6%**. *(Thesis §2.4, Table 2.9)*

- [ ] **5.17** A detector for the error the system actually makes  *(10 s)*

```bash
$PY scripts/signal_rebuild/run_negative_evidence_detector.py --verify
```

**Reproduces:** explicit negative prompts (printed text, sheet music, ornament, blank page, title page, table, map) scored with SigLIP's own logit scale and bias. ROC-AUC vs. scene presence: deployed **0.4345**, corrected single channel **0.7188**, negative evidence alone **0.8140**, margin **0.8295**, fitted on both **0.8305**. Against the *corrected* baseline: **+0.111, CI [+0.045, +0.181]** — the only interval in this thesis that excludes zero. Test error rate **29.7%**; flags **81.6%** of true negatives against **31.8%** of positives. *(Thesis §2.4, Table 2.10)*

**Channel switching.** Five scripts now take `--channel {softmax,sigmoid}`. The default `softmax` reproduces the deployed record (all committed JSONs); `--channel sigmoid` writes to `<name>_corrected.json`:

```bash
for f in run_redesigned_signals diagnose_why_disagreement_fails \
         run_redesigned_full_corpus_and_recommend run_fixed_signals rebuild_from_full_tier1; do
  $PY scripts/signal_rebuild/$f.py --channel sigmoid
done
```

- [ ] **5.18** Can the formulation itself be fixed?  *(75 s)*

```bash
$PY scripts/signal_rebuild/run_formulation_ceiling_corrected.py --verify
```

**Reproduces:** the formulation *can* be fixed, but not by repairing `A_agree` or `A_vlm`. Replacing `A_vlm` with a term measuring a scene's **absence** gives `SAA = 0.192·A_object + 0.500·A_scene − 0.308·A_negative`, raising scene ranking **0.7599 → 0.8231** (+0.0631, CI [+0.0208, +0.1138], excludes zero) and balanced accuracy 0.6873 → 0.7115.

Also reproduces, against the premise that `A_vlm` does nothing: it is near chance at ranking **errors** (0.569), but deleting it from the **fusion** raises the error rate 36.6% → 39.8% and lowers scene ranking 0.7599 → 0.7408. The two levels ask different questions of the same signal.

And the ceiling for the continuous case, where `8!` enumeration is impossible because `A_scene` has 1,520 values: every unsupervised operator loses badly (std **0.2310**, range 0.2320, mean pairwise 0.2052, BALD **0.4727**) and flexible models fitted *inside* the test partition — held out only by publication fold, so optimistic — reach only **0.8251–0.8391** against **0.8480** for plain confidence. One candidate comes close (gradient boosting over confidence + 5 repaired levels, **0.8922**, stable to sd 0.0002 over 15 seeds) but its multiplicity-corrected interval straddles zero and it is AWLF's supervised route, not a disagreement statistic. *(Thesis §2.4, Table 2.11)*

- [ ] **5.19** Does the risk guarantee hide its own failure?  *(2 s)*

```bash
$PY scripts/signal_rebuild/run_conditional_risk_control.py --verify
```

**Reproduces:** the deployed guarantee controls the **marginal** risk, averaged over a pool that is 84.5% scene-positive. Decomposed by predicted class, at the reported operating point α = 0.05 the overall bound is satisfied (risk 0.0482) while the **predicted-positive group carries risk 0.0652 — 30% above the same target — with 100% of that group auto-approved**. At α = 0.01 the same group exceeds by 45%. At α ≥ 0.10 the pattern inverts and the negative group fails badly (0.3694 against 0.15).

The Mondrian (class-conditional) reformulation is implemented and holds at 3 of 5 levels, failing at the same two. The cause is sample size, not the formulation: the test partition holds 138 predicted-positive images carrying 9 errors and 111 predicted-negative carrying 82. Conditioning also costs 1.6–15.3 pp of automatic approval. *(Thesis §2.4, Table 2.12, Eq. 2.x)*

- [ ] **5.10** Limitation probes  *(62 s)*

```bash
$PY scripts/core_pipeline/run_limitation_probe_experiments.py --verify
```

**Reproduces:** the four empirically decidable limitations (backup slide A.16). *(Thesis §7.4)*

---

## Stage 6 — The fusion weights and the knife-edge  *(~1 s)*

- [ ] **6.1** Weight sensitivity  *(1 s)*

```bash
$PY scripts/core_pipeline/run_formulation_sensitivity.py --verify
```

**Reproduces:** Table 4.2 — deployed weights **70** errors, every alternative **56**. *(Thesis §4.2.1)*

- [ ] **6.2** Can the weights be learned?  *(<1 s)*

```bash
$PY scripts/signal_rebuild/run_learned_fusion_weights.py --verify
```

**Reproduces:** only **6** reachable classifiers (15 with a free threshold); fitting by error count returns the degenerate all-positive rule; the knife-edge cell `(⅓, 0.5, 1)` holds **276** labelled images and computes to **0.4999998** — which is the entire 70-vs-56 difference. *(Thesis §4.2.1)*

---

## Stage 7 — The shared blind spot on true negatives  *(~35 s)*

- [ ] **7.1** The expanded negative-control pool  *(14 s)*

```bash
$PY scripts/annotation/evaluate_expanded_negative_controls.py --verify
```

**Reproduces:** blind spot **0.1259** on n=319 confirmed negatives, 50 errors, CI [0.0619, 0.1959]; replicated at **0.1852** on 123 independent new negatives. *(Thesis §4.3)*

- [ ] **7.2** Why the value is below chance  *(<1 s)*

```bash
$PY scripts/signal_rebuild/decompose_negative_control_blind_spot.py --verify
```

**Reproduces:** 115 of the 319 negatives are error-free by arithmetic (`A_object`=0 ⇒ `SAA` ≤ 0.2); within the stratum where errors are possible the figure rises to **0.22**. *(Thesis §4.3)*

- [ ] **7.3** A complementary detector  *(20 s)*

```bash
$PY scripts/signal_rebuild/run_unanimous_fp_detector.py --verify
```

**Reproduces:** a detector built from raw signals that never enter `SAA` reaches **0.7616**, but does not separate from confidence (+0.0148, CI [−0.074, 0.096]). *(Thesis §4.3)*

---

## Stage 8 — Tier 2 agents against human labels  *(<1 s)*

- [ ] **8.1** Gold-labelled precision of the curation agents

```bash
$PY scripts/core_pipeline/run_tier2_agent_gold_precision.py --verify
```

**Reproduces:** Agent 2 contradiction detection **23.3%** precise (n=101); Agent 4 cross-archive linking **55.7%** (n=79), 100% at similarity ≥ 0.92; Agent 5 grounded OCR **52.5%** (n=53). *(Thesis §6)*

---

## Stage 9 — The pipeline rebuilt on real inference  *(~40 s)*

These consume the Stage 1 outputs.

- [ ] **9.1** Rebuild all four signals from the full Tier 1 run  *(9 s)*

```bash
$PY scripts/signal_rebuild/rebuild_from_full_tier1.py --verify
```

**Reproduces:** the deployed `A_vlm` rule is **2/4 on 98.9%** of images; rebuilt `U_SAA` takes **11,786** distinct values instead of 8; dependency 100% → 78.9%; disagreement **0.8098** vs confidence **0.8483**. *(Thesis §2.4)*

- [ ] **9.2** A real detector on every image  *(13 s)*

```bash
$PY scripts/signal_rebuild/run_real_detector_retest.py --verify
```

**Reproduces:** real detector vs fallback correlate at r = 0.59, agree exactly on 25.7%; disagreement 0.8304 vs confidence 0.8674; the continuous variant is flagged **degenerate** (82% error rate). *(Thesis §2.4)*

- [ ] **9.3** The rebuilt pipeline end to end, recalibrated  *(1 s)*

```bash
$PY scripts/signal_rebuild/run_end_to_end_rebuilt.py --verify
```

**Reproduces:** inherited 0.5 threshold → 43.4% error; retuned on calibration → **27.7%**; recalibrated CRC at α = 0.05 → **55.8%** auto-approval at **84.1%** recall, bound met; 53,000-image projection **~21 GPU-hours**. *(Thesis §2.4)*

- [ ] **9.4** Carry it through Tier 2 and Tier 3  *(6 s)*

```bash
$PY scripts/signal_rebuild/run_tier2_tier3_from_full_tier1.py --verify
```

**Reproduces:** Agent 2 fires on **71.97%** of the corpus; flags land on **85.0%** of auto-approved vs **41.4%** of escalated images; Tier 3 load **3,620** images. *(Thesis §2.4)*

- [ ] **9.5** The decision pipeline at archive scale  *(11 s)*

```bash
$PY scripts/signal_rebuild/run_full_corpus_deployment.py --verify
```

**Reproduces:** corpus auto-approval **70.1%** (8,490) vs 60.6% on the test partition; labelled pool representative, TVD **0.076**; λ̂ bootstrap CI [0.372, 0.372], never crossing the knife-edge cell. *(Thesis §5.3)*

---

## Stage 10 — Compile every document  *(~3 min)*

- [ ] **10.1** Compile all five documents

```bash
cd presentation
for f in thesis_v3_updated defense_slides defense_handout formula_derivations advisor_meeting_2026-09-14; do
  ../tectonic -X compile $f.tex > /dev/null 2> /tmp/$f.log \
    && echo "OK    $f.pdf" || echo "FAIL  $f  (see /tmp/$f.log)"
done
cd ..
```

Expected: five `OK` lines. Page counts: thesis 118, defence deck 60, handout 2, formula reference 18, advisor deck 33.

- [ ] **10.2** Check for compile errors

```bash
grep -icE '^error|! LaTeX|! Undefined' /tmp/thesis_v3_updated.log   # expect: 0
```

---

## Stage 11 — Everything at once

When you trust the individual stages, this runs every verification in one go and prints a scorecard.

- [ ] **11.1** Run the full verification sweep  *(~10 min)*

```bash
cd /data/brhanu/thesis_project
PY=.venv/bin/python3
pass=0; fail=0
for f in $(grep -rln '"--verify"' scripts/ --include="*.py" | grep -v archive | sort); do
  out=$($PY "$f" --verify 2>&1 | tail -1)
  case "$out" in
    *"ALL FIGURES REPRODUCE"*|*IDENTICAL*) echo "PASS  $(basename $f)"; pass=$((pass+1));;
    *)                                     echo "FAIL  $(basename $f)  ::  $out"; fail=$((fail+1));;
  esac
done
echo; echo "=== $pass passed, $fail failed ==="
```

Expected: **`=== 37 passed, 0 failed ===`**

- [ ] **11.2** Run it in the background, so it survives closing the terminal

```bash
nohup bash -c 'cd /data/brhanu/thesis_project; PY=.venv/bin/python3
for f in $(grep -rln "\"--verify\"" scripts/ --include="*.py" | grep -v archive | sort); do
  echo "$($PY "$f" --verify 2>&1 | tail -1)  <-  $(basename $f)"
done' > verify_sweep.log 2>&1 &

tail -f verify_sweep.log          # Ctrl+C to stop watching
```

---

## Stage 12 — Annotation status and the Label Studio queue

The evaluation pool is the binding constraint on every certified number: the test partition
holds **249 images and 38 true negatives**, and the negative class is what the
negative-evidence detector (5.17) exists to find.

**Current state of task A2** (the remaining negative controls):

| | count |
|---|---|
| A2 images total | 750 |
| labelled so far | **285** (43.2% `no_scene`) |
| remaining | **465** |
| **already merged into the evaluation split** | **0** |

The 285 labels already collected are **not yet in the split** — the current 1,250-row pool
is built from the earlier CVAT and task-A passes only.

- [ ] **12.1** Check the queue state

```bash
DJANGO_SETTINGS_MODULE=core.settings.label_studio \
/data/brhanu/miniconda/bin/python3.13 \
/data/brhanu/miniconda/lib/python3.13/site-packages/label_studio/manage.py shell -c "
from projects.models import Project; from tasks.models import Task, Annotation
p = Project.objects.get(id=8)
t = Task.objects.filter(project=p).count()
a = Annotation.objects.filter(task__project=p).count()
print(f'tasks {t}  annotated {a}  remaining {t-a}')"
```

- [ ] **12.2** Label in the browser

Label Studio runs on **:8080**, images are served CORS-enabled from **:9000**. Both are
long-running processes; restart either with:

```bash
label-studio start --no-browser --port 8080          # if not already running
$PY scripts/core_pipeline/serve_images_cors.py 9000  # serves cvat_upload_temp/obj_train_data
```

Then open `http://localhost:8080` → project **"Self-Annotation A2: Remaining Negative
Controls (750)"** (id 8) → **Label All Tasks**. Only unlabelled tasks are served.

**What completing it buys** (projected at the observed 43.2% negative rate and the current
60/20/20 split):

| | now | projected |
|---|---|---|
| labelled pool | 1,250 | 2,000 |
| test partition | 249 | ~399 |
| true negatives in test | **38** | **~103** |

⚠️ **Merging changes the group-disjoint split, so every certified figure is recomputed.**
That is mechanical — all 35 experiments carry `--verify` and re-run in about ten minutes —
but it is a deliberate re-baselining, not an additive patch. Do it once, after annotation
completes, not incrementally.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MISMATCH` from a script | An input changed since the committed result was produced | Run `git status` and `git diff results/` to see what changed. If you changed an input on purpose, run the script without `--verify` to regenerate, then commit. |
| `FileNotFoundError` on a `results/signal_rebuild/*` file | Stage 1 outputs are missing | Run Stage 1, or restore the outputs from git. |
| `CUDA out of memory` in Stage 1 | Another process is holding GPU memory | Check `nvidia-smi`; stop the other process. LLaVA needs ~44 GB per GPU. |
| LLaVA: `prompt longer than the maximum model length` | Full-resolution scans produce too many image tokens | Already handled — images are capped at 640 px and the context is 16,384. Do not remove either. |
| Stage 1 stopped halfway | Session ended, or a crash | Re-run step 1.1. Every model resumes from where it stopped. |
| `ModuleNotFoundError: saa_decision` | Script run from the wrong directory | `cd /data/brhanu/thesis_project` first. |
| A timing loop prints `printf: invalid number` | The system locale uses a decimal comma | Harmless for the verification itself. Prefix with `LC_ALL=C` if it bothers you. |

---

## What these results rest on — keep this in mind when reading them

- **Human labels cover 13.0% of the corpus** (1,580 of 12,110). Every accuracy figure is computed on that subset — most on the 249-image test partition.
- **The headline CRC risk (0.0201) rests on 5 errors** among 151 auto-approved images. Exact 95% CI [0.013, 0.071]. The auto-approved audit in `self_annotation_package/task_e_auto_approved_audit/` exists to tighten this.
- **Two decision rules coexist.** Scripts reading the stored `comparison_fusion_score` use `STORED_COLUMN_THRESHOLD`; scripts recomputing `SAA` use `SAA_THRESHOLD`. They disagree on 2,217 images. Both are named constants in `scripts/core_pipeline/saa_decision.py`, deliberately not unified — see Thesis §4.2.1.
- **Stage 1 uses a different detector checkpoint** from the historical one. The distilled `student_iter_3` checkpoint is no longer on disk; Stage 1 uses the trained YOLOv11m archival-taxonomy model at `runs/detect/yolo11_v2_augmented_refresh`.

---

*25 gated scripts. Every one verified reproducing on 11 September 2026.*
