# Full Project Re-Run and Analysis

**Run date:** 13 September 2026 · **Verification sweep:** 36 passed, 0 failed, 12 m 17 s · **Documents:** 5 rebuilt from a clean tree, all text-identical to the committed PDFs

---

## 1. What was re-run, and what was not

| Stage | Status | Notes |
|---|---|---|
| Tier 1 model inference (4 models × 12,110 images) | **not re-run** | see below |
| All 36 derived experiments | **re-run, all pass** | 12 m 17 s total |
| All 5 documents, clean tree | **rebuilt** | text-identical to committed |
| Raw output integrity | **checked** | complete, no gaps |

**Why Tier 1 inference was not re-run.** All four A6000s are occupied by two other researchers' jobs (`ghosh`, `bezer`), leaving 6–17 GB free per card. Loading LLaVA-OneVision and Kosmos-2.5 would have risked an out-of-memory failure in their training runs. This is a scheduling constraint, not a technical one: the run is scripted, resumable, and takes about 4 h 40 m on four free cards.

**What was checked instead.** The stored raw outputs are complete and consistent: 12,110 rows and 12,110 unique image IDs for each of the four models, with no duplicates and no missing pages.

| Model | Rows | Unique images | Size |
|---|---|---|---|
| YOLO detector | 12,110 | 12,110 | 2.1 MB |
| LLaVA (4 shards) | 12,110 | 12,110 | 11.3 MB |
| Kosmos OCR (4 shards) | 12,110 | 12,110 | 14.3 MB |
| SigLIP scene | 12,110 | 12,110 | 4.2 MB |
| SigLIP embeddings | — | 12,110 | 47.1 MB |

**Scale of what was verified:** 130 result files, 72 CSV tables, **875,066 individual numeric values**.

---

## 2. The central claim, across every test ever run

Thirty formulations have now been compared against the same baseline — a plain confidence score with no disagreement term.

| Outcome | Count |
|---|---|
| Beat confidence with an interval excluding zero | **1** |
| Significantly **worse** than confidence | **19** |
| Worse, not significantly | 6 |
| Bounds, not comparisons (the two ceilings) | 4 |

The single apparent winner is a gradient-boosted model over confidence **plus** the five repaired signal levels: +0.0438, uncorrected CI [+0.0103, +0.0797]. Corrected for the nine formulations compared against that baseline, the interval becomes [−0.0012, +0.0946] — **it straddles zero, and the lower bound changes sign between bootstrap runs.** It is also not a disagreement statistic: it uses the signal *levels* under supervision, which is the route AWLF already takes.

### The two ceilings

- **Discrete case.** The deployed signals reach only 8 distinct states. Enumerating all 8! = **40,320** orderings gives a best possible 0.8871 against 0.8828 for confidence — a maximum possible gain of **0.0043**.
- **Continuous case.** On the corrected channel `A_scene` takes 1,520 values, so enumeration is impossible. Flexible models fitted *inside* the test partition — held out only by publication fold, which is optimistic because model selection sees the distribution — reach 0.8251–0.8391 against **0.8480** for confidence. An optimistic upper bound that still loses.

### Why it fails, mechanically

| Measurement | Deployed | Corrected |
|---|---|---|
| Effective rank of 4 signals | 1.677 | 1.845 |
| All models wrong together | 6.2 % | 10.7 % |
| Spread AUROC within SAA quartiles | 0.34–0.75 | 0.37–0.92 |
| Mean spread: errors vs correct | 0.4307 / 0.4178 | 0.3960 / 0.3909 |
| Mean level: errors vs correct | 0.3347 / 0.6242 | 0.1543 / 0.4942 |

The level separates errors from correct by **0.339**; the spread separates them by **0.005**. Error is defined by the level crossing a threshold, and a standard deviation is a second moment built to ignore the level.

### The unanimity result

Every calibrated member sat above 0.5 on every test image — the ensemble was unanimous on **100 %** of pages, a consequence of the 84.5 % positive base rate. Class-balanced calibration removes it:

| Configuration | Unanimity | Disagreement AUROC |
|---|---|---|
| deployed channel, unweighted | 100 % | 0.612 |
| deployed channel, balanced | 11.2 % | 0.589 |
| corrected channel, unweighted | 100 % | 0.647 |
| **corrected channel, balanced** | **14.9 %** | **0.5014** |

In the one configuration where all four models genuinely reach different conclusions, disagreement predicts errors at **chance**.

---

## 3. What works — the three positive results

All three intervals exclude zero.

| Change | Before | After | Δ | 95 % CI |
|---|---|---|---|---|
| Scene channel: calibrated output, not forced choice | 0.4172 | **0.7754** | +0.3578 | [+0.2991, +0.4145] |
| Fusion: add a term for a scene's **absence** | 0.7599 | **0.8231** | +0.0631 | [+0.0208, +0.1138] |
| Negative-evidence detector vs corrected baseline | 0.7188 | **0.8305** | +0.1120 | [+0.0443, +0.1828] |

End to end:

| Configuration | Error rate | Balanced accuracy |
|---|---|---|
| deployed channel, hand-set weights | 42.2 % | 0.6649 |
| deployed channel, refitted | 37.8 % | 0.6909 |
| corrected channel, hand-set | 39.4 % | 0.6815 |
| corrected channel, refitted | 36.5 % | 0.6873 |
| **negative-evidence detector alone** | **29.7 %** | **0.7491** |

---

## 4. A finding this re-run surfaced: the risk bound is more fragile than reported

Counting how often the conformal promise actually held across five risk levels:

| Configuration | Bound held |
|---|---|
| deployed pipeline, as reported in the thesis | 5 / 5 |
| deployed channel, hand-set weights | 3 / 5 |
| deployed channel, refitted | 4 / 5 |
| corrected channel, hand-set | 2 / 5 |
| corrected channel, refitted | 2 / 5 |

Each configuration defines its own error label, so these are not strictly comparable to one another. But the pattern is real and should be stated: **improving the classifier does not automatically improve the guarantee**, and on a 249-image test partition the bound is missed often enough that a single headline "the guarantee holds" overstates it. Conformal risk control bounds risk *in expectation over the calibration draw*, not on every realisation — with this few errors, exceedances are expected.

This is the strongest argument for more labelled pages in the whole project.

---

## 5. Everything measured against human labels

| Claim | Measured | Verdict |
|---|---|---|
| Ranks errors on scene-positive pages | 0.9687 held out | **works** |
| Ranks errors on confirmed blank pages | 0.1259 (n = 319) | **fails, below chance** |
| Detection vs human boxes | mAP50 0.1462 (n = 300) | **weak** |
| Detection vs its own teacher | mAP50 0.989 | the paradox |
| Calibration, same cohort | ECE < 0.001 | **works** |
| Calibration, new publications | ECE 0.109 | **does not transfer** |
| Agent 2 contradictions | 23.3 % precise | aid only |
| Agent 4 cross-archive links | 55.7 %, 100 % above 0.92 | usable with a threshold |
| Agent 5 OCR | 52.5 %, median WER 100 % | aid only |

**Corpus scale.** Routing on the spread certifies 10.0 % of the archive for automatic approval (18.5 % on the corrected channel); routing on confidence certifies **61.1 %** (64.6 % corrected), at the same risk level and with higher error recall. Tier 2's contradiction agent fires on 71.97 % of the corpus — and more often on the *auto-approved* pile (85.0 %) than the escalated one (41.4 %), so it supplies no corroboration for the triage decision.

---

## 6. The four signals, audited

| Signal | Defect | Fixed? |
|---|---|---|
| `A_object` | not the detector — a fallback VQA count, capped at 3 | real detector now run corpus-wide |
| `A_vlm` | identical value on **98.92 %** of images | rebuilt; still near chance |
| `A_agree` | a tent function of one other signal on **98.92 %**; correlation only 0.18, so exact but non-linear — correlation screens passed it | rebuilt; **repairing made it worse** |
| `A_ocr` | saturates at maximum on **68.6 %** of pages, counting characters not sense | rebuilt; works as validity, irrelevant to scene |
| `A_scene` | read from the forced-choice output instead of the calibrated one — **below chance at its own task** | **fixed; the largest single gain in the project** |

---

## 7. Honest assessment

**What is solid.** The routing mechanism, the reproducibility, the corpus-scale run, and the negative result itself. Thirty formulations and two ceilings is not an inconclusive comparison — it is a bound.

**What is thin.** Everything rests on **249 test images containing 38 blank pages**. That is why the best new result cannot be proven as firmly as it deserves, why the risk bound misses at some levels, and why the blind-spot finding moved twice as labels were corrected.

**What is overclaimed.** The title promises object detection and complex scene detection. The evidence supports an object *count* and a *binary* scene decision. Against human-drawn boxes the detector reaches 0.1462 mAP50, and no multi-class scene result can be reported — one category holds three-quarters of the labels and another holds seven.

**The single highest-value action** is not compute. It is the 465 prepared, unlabelled pages.

---

## Correction, 14 September

An earlier version of this report listed "10-class image-level detection: macro F1 0.528, 2 classes at exactly 0" and said two of ten object classes had no measurable performance. **Those figures were not real.** They came from `results/multi_agent/gold_subset_per_class_metrics.csv`, which was computed on 10 June from `human_baseline_gold_kit/labeling_worksheet.csv` — the folder later found to contain fabricated annotation and deleted. The derived CSV survived on disk and was picked up by the analysis. Both mentions have been removed. The only human-referenced detection figure is mAP50 = 0.1462 on the 300 pictures with human-drawn boxes.

