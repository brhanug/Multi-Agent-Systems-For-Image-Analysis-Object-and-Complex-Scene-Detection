# Reproducing the results in this thesis

The thesis is `presentation/thesis_v4.tex`. Which script backs which chapter is in
[README.md](README.md), section 2. This file says how to rerun them and how to check the result.

Last verified: **20 November 2026.** The four checkers report 723 of 723 cited numbers matching
their result files, and the headline chain below reproduces exactly.

---

## 1. Check that the thesis matches the results (one minute, no GPU)

```bash
cd /data/brhanu/thesis_project
P=.venv/bin/python3
$P scripts/thesis_v4_chapter6_check.py        # expect: 586 of 586 ... / 55 of 55 ...
$P scripts/thesis_c1_figures_check.py         # expect: 43 of 43 ...
$P scripts/thesis_rerun_figures.py --verify   # expect: 39 of 39 ...
```

Nothing is rewritten. A mismatch names the sentence and the result file.

---

## 2. Rerun from cached model outputs (minutes, no GPU)

```bash
bash scripts/run_full_master_experiments.sh             # deployed design, Chapters 2–5 and 8
for f in scripts/signal_rebuild/run_*.py; do .venv/bin/python3 "$f"; done   # corrected design
```

The second line is what step 09 of `scripts/rerun_everything_2026-11.sh` does, in the order the
scripts were added. Use that script if the order matters.

### The deployed design's headline chain (about 3 seconds)

```bash
P=.venv/bin/python3
$P scripts/core_pipeline/build_corrected_negative_control_pool.py --verify
$P scripts/core_pipeline/build_corrected_clean_holdout_split.py
$P scripts/core_pipeline/run_corrected_clean_holdout_evaluation.py
$P scripts/core_pipeline/run_corrected_clean_holdout_crc.py
$P scripts/core_pipeline/run_tier2_agent_gold_precision.py --verify
$P scripts/core_pipeline/run_europeana_transfer_ground_truth_eval.py
```

| Script | What it should print |
|---|---|
| `build_corrected_negative_control_pool.py --verify` | `1250 rows \| negatives 196` and `IDENTICAL` |
| `build_corrected_clean_holdout_split.py` | `zero publication (PPN) overlap`, split 749 / 252 / 249 |
| `run_corrected_clean_holdout_evaluation.py` | `Multi-Agent SAA (Ours) AUROC=0.8600`, `AWLF ... AUROC=0.8702` |
| `run_corrected_clean_holdout_crc.py` | `Alpha=0.05 ... TestRisk=0.0201 ... AutoApprove=60.6% Recall=91.1% Lift=2.31x` |
| `run_tier2_agent_gold_precision.py --verify` | `ALL FIGURES REPRODUCE` |
| `run_europeana_transfer_ground_truth_eval.py` | `65.2% false negatives`, `ensemble degenerate: True` |

---

## 3. Full rerun, including model inference (hours, GPUs)

```bash
nohup bash scripts/rerun_everything_2026-11.sh &
```

It reruns LLaVA, SigLIP, the detectors, Kosmos-2.5 and the Grounding DINO teacher over all
12,110 pictures, retrains the student detector, reruns every experiment, rebuilds the thesis and
runs the checkers. It is resumable: finished steps are recorded under `reruns/rerun_2026-11-03/state/`
and skipped on restart. Before it starts, it backs up `results/` to `reruns/.../backup/`.

The last full rerun (3 November 2026) reproduced 71 of 72 experiments. The LLaVA, Kosmos and
retrained-student figures moved slightly, because GPU inference is not bit-exact.

---

## 4. Re-extracting the hand labels

Only needed if labelling continues in Label Studio. The Label Studio database is at
`~/.local/share/label-studio/label_studio.sqlite3`.

```bash
.venv/bin/python3 scripts/core_pipeline/extract_self_annotation_labelstudio.py   # tasks A–D
.venv/bin/python3 scripts/core_pipeline/extract_europeana_annotations.py         # Europeana pilot
```

The quick-label and archive-class tasks write their CSVs directly, under `self_annotation_package/`.

---

## 5. What cannot be regenerated

These are human labour. If they are lost, they must be labelled again by hand. Back them up.

- `human_spatial_audit/user_annotations_800.csv`: the 801-picture expert cohort (C1)
- `human_spatial_audit/user_annotations_300.csv`: the 300-picture box audit (C2)
- `results/multi_agent/negative_control_reverification_audit.json`: the 144-picture re-audit
- `self_annotation_package/task_*/labeling_sheet_LS_export.csv`: the gold sets of tasks A–D
- `self_annotation_package/{quick_labels,final_check,confirm_check,archive_classes,class_tasks,scene_consistency}/`: the labels behind Chapters 6 and 7
- `results/external_archive/europeana_real_pilot/europeana_human_annotations_REAL.csv`
- the Label Studio database

`.gitignore` keeps exceptions so these stay in git even though `results/` is ignored.
**Do not "clean up" those exceptions.**

---

## 6. Ground rules

1. No script writes numbers into a `.tex` file. Results go to `results/`; the text cites them,
   and the checkers in section 1 hold the two together.
2. Every cited number needs a script that regenerates it and a checker entry that pins it.
3. Prefer `--verify` over trust: recompute, compare with the stored result, fail loudly on drift.
4. `scratch/calibrate_temporal.py` and `scratch/validate_temporal_agent.py` contain synthetic
   data (`1850 + hash(id) % 100`). Never cite their output.
5. An image's page id is not unique across books. Join expert labels to pictures with
   `scripts/core_pipeline/picture_ids.py`.

---

## 7. Building the documents

```bash
cd presentation
~/.local/bin/tectonic -X compile thesis_v4.tex
~/.local/bin/tectonic -X compile defense_slides_v2.tex
~/.local/bin/tectonic -X compile defense_handout_v2.tex
```

Expected: no errors, no undefined references, no missing citations.
