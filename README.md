# Multi-agent systems for image analysis: object and complex scene detection

**When Can an Automatic Answer Be Trusted? Confidence, Not Disagreement, Under a Risk Promise for Historical Illustrations**

Master thesis, Brhanu Atsbaha, University of Hildesheim. Advisors: Prof. Dr. Thomas Mandl and Shereen Elsayed.
Submitted December 2026; defence on 10 December 2026. The submitted PDF is [presentation/thesis_v4.pdf](presentation/thesis_v4.pdf) (142 pages).

This repository holds the thesis, the defence material, and every script behind a number in them.
The system the thesis puts forward is the **corrected design** of Chapters 6 and 7: three agents
(an object finder, a scene matcher and a describer), each turned into a calibrated probability,
combined by a learned mapping, and routed under a conformal risk-control promise. Chapters 3 to 5
describe the earlier, deployed design and why it was replaced.

---

## 1. The documents

| Document | Source | Build |
|---|---|---|
| Thesis ([PDF](presentation/thesis_v4.pdf)) | [presentation/thesis_v4.tex](presentation/thesis_v4.tex) (a copy of `thesis_v3_updated.tex`, which is the file to edit), which includes [thesis_corrected_design_chapter.tex](presentation/thesis_corrected_design_chapter.tex) (Ch. 6–7), [thesis_corrected_design_appendix.tex](presentation/thesis_corrected_design_appendix.tex) (App. G) and [latex_assets/failure_gallery.tex](latex_assets/failure_gallery.tex) (App. A) | `cd presentation && cp thesis_v3_updated.tex thesis_v4.tex && ~/.local/bin/tectonic -X compile thesis_v4.tex` |
| Defence slides, Beamer (48 pages) | [presentation/defense_slides_v2.tex](presentation/defense_slides_v2.tex) | same, with `defense_slides_v2.tex` |
| Defence slides for the talk (26 slides and 25 backup slides) | [presentation/defence_deck_src/](presentation/defence_deck_src/) (the HTML slides of the online deck) | export a PDF from the deck page |
| Defence talk, Beamer (16:9, 38 talk slides + backup; every model's input and output, definitions, diagrams, one picture followed through) | [presentation/defence_talk.tex](presentation/defence_talk.tex) | `tectonic -X compile defence_talk.tex` (example pictures in `presentation/defence_assets/`, not in git) |
| Defence preparation: likely questions and the numbers to quote | [presentation/defense_prep_sheet.md](presentation/defense_prep_sheet.md) | |
| Handout (2 pages) | [presentation/defense_handout_v2.tex](presentation/defense_handout_v2.tex) | same, with `defense_handout_v2.tex` |

Figures live in [latex_assets/](latex_assets/), the bibliography in [presentation/references.bib](presentation/references.bib).
There is no system LaTeX on the server; use Tectonic.

---

## 2. Chapter → script → result

The scripts for the corrected design are in [scripts/signal_rebuild/](scripts/signal_rebuild/), and
they follow one rule: **`scripts/signal_rebuild/run_X.py` writes `results/signal_rebuild/X.json`**.
The scripts for the deployed design (Chapters 2–5 and 8) are in [scripts/core_pipeline/](scripts/core_pipeline/)
and write to `results/multi_agent/`.

### Main text

| Ch. | Section | Scripts (`run_` prefix omitted where the rule above applies) |
|---|---|---|
| 1 | Introduction, state of the art, baselines | text only |
| 2 | Method and architecture, the data, restoration, SAA formula | core: `run_agent_comparison.py`, `run_multi_agent_validation.py`, `run_scene_complexity_index.py` |
| 3 | Spatial grounding collapse (0.768 → 0.146) | core: `run_spatial_error_prediction_100.py`, `run_box_level_iou_uncertainty.py`, `run_iou_error_taxonomy.py`; signal_rebuild: `run_ultralytics_val_deployed.py` (0.146 with the same validator as 0.768; `--device N` picks the GPU); labels in `human_spatial_audit/user_annotations_300.csv` |
| 4 | Disagreement as uncertainty, held-out validation, Platt calibration, Tier 1 ablation | core: `build_corrected_negative_control_pool.py`, `build_corrected_clean_holdout_split.py`, `run_corrected_clean_holdout_evaluation.py`, `run_uncertainty_signal_decomposition.py`, `run_calibration_metrics.py`, `run_tier1_ablation.py` |
| 5 | Triage and conformal risk control (deployed design) | core: `run_corrected_clean_holdout_crc.py`, `exp29_conformal_risk_control.py`, `run_hitl_simulation.py` |
| 6 | **The corrected design** | see the table below |
| 7 | **The corrected design on the archive** | see the table below |
| 8 | Agents 2, 4, 5 and the retrieval pilot | core: `run_tier2_agent_gold_precision.py`, `check_agent5_ocr_degeneracy.py`, `run_semantic_retrieval_eval.py`; signal_rebuild: `run_agent2_on_300.py`, `run_agent4_link_checks.py`, `run_agent5_buckets_rerun.py`, `run_human_judgements.py` (retrieval and archive links judged by a person) |
| 9 | Conclusions, ethics, what the thesis does not claim, limitations | text only |

### Chapters 6 and 7, section by section

All in `scripts/signal_rebuild/`. Each row was derived from [scripts/thesis_v4_chapter6_check.py](scripts/thesis_v4_chapter6_check.py),
which ties every number in these sections to its result file.

| Section | Scripts |
|---|---|
| 6.2 Three agents, each reporting what it measures | `run_per_model_performance_300.py`, `run_worked_example.py` |
| 6.3 From each agent's report to a probability | `run_worked_example.py` (running example: `--picture PPN1756648816/00000272_0 --out worked_example_family.json`) |
| 6.5 The risk promise | `run_corrected_tier1_end_to_end.py`, `run_crc_curve_figure.py`, `run_per_batch_risk.py`, `run_example_pictures.py` |
| 6.6 Evaluation protocol | `run_corrected_resampled_t.py` |
| 6.7 The five scene classes | `run_scene_classes.py`, `run_scene_classes_joint.py` (the system: learned joint mapping), `run_combination_weights.py`, `run_combination_selection.py`, `run_combination_c_extra.py`, `run_more_experiments.py` |
| 6.8 The eight object classes | `run_object_classes.py`, `run_object_promise_detail.py`, `run_per_class_risk_control.py`, `run_prompt_pilot_overlap.py` |
| 6.9 Scene presence on 200 redraws | `run_best_config_repeated.py`, `run_ablation_corrected.py`, `run_presence_joint.py`, `run_learned_old_design.py`, `run_triage_weight.py`, `run_label_shift_check.py`, `run_out_of_pool_check.py` |
| 6.10 The decision line | `run_corrected_calibration.py`, `run_decision_threshold_check.py`, `run_two_sided_risk_control.py` |
| 6.11 The risk promise at matched realised risk | `run_matched_risk_comparison.py`, `run_per_batch_risk.py` |
| 6.12 Review effort for the archive | `run_review_hours.py` |
| 6.13 The system against its baselines | `run_baselines.py`, `run_baselines_extra.py` |
| 7.1 Does the promise hold on the archive? | `run_final_check.py` (704 pictures), `run_confirm_check.py` (300 pictures), `run_archive_approval_breakdown.py`, `run_external_audit.py`, `run_presort_controls.py` |
| 7.2 The system's answers across the archive | `run_archive_answers.py`, `run_archive_scene_classes.py` (`--partial`: the 179 of 444 pictures labelled) |
| 7.3 A better teacher for the object finder | `run_teacher_candidates_on_300.py`, `train_gdino_student.py` |
| 7.4 A pre-registered test of disagreement | `run_c4_preregistered.py`, `run_c4_task_aligned.py` (disagreement within each task) |
| 7.5 Scene agreement: systems and people | `run_scene_agreement.py` |
| 7.6 Does restoration help? | `run_restoration_ablation.py` |

### Appendices

| App. | Content | Scripts |
|---|---|---|
| A | Example pictures | signal_rebuild: `run_example_pictures.py` |
| B | PeopleArt cross-style benchmark | core: `run_peopleart_evaluation_real.py`, `generate_peopleart_real_plots.py` |
| C | Europeana transfer pilot | core: `run_europeana_transfer_ground_truth_eval.py`, `run_ensemble_heterogeneity_diagnostic.py` |
| D | Derivation of the SAA signals | `run_worked_example.py` |
| E | Deployed signals: audit and repair | signal_rebuild: `rebuild_from_full_tier1.py`, `run_redesigned_signals.py`, `run_fixed_signals.py`, `diagnose_why_disagreement_fails.py` |
| F | Detailed analyses of the deployed design | core (sections 1–17 of `run_full_master_experiments.sh`), signal_rebuild: `run_formulation_search.py`, `run_fusion_weights_corrected.py`, `run_label_filter_experiment.py` |
| G | Further analyses of the corrected design | `run_scene_definition_check.py`, `run_decision_threshold_check.py`, `run_label_filter_experiment.py`, `run_calibrated_agreement.py`, `run_blind_spot_features.py` |

---

## 3. Checking that the thesis matches the results

Three checkers compare cited numbers with their result files. None of them edits the thesis.

```bash
P=.venv/bin/python3
$P scripts/thesis_v4_chapter6_check.py        # Chapters 6–7, App. A and G: 662 numbers; scene-class and other main-file figures: 109
$P scripts/thesis_c1_figures_check.py         # in-sample (C1) figures: 43 numbers
$P scripts/thesis_rerun_figures.py --verify   # figures that predate v4: 39 numbers
```

State on 25 November 2026: **853 of 853 match** (662 + 109 + 43 + 39).

---

## 4. Reproducing the results

See [REPRODUCE.md](REPRODUCE.md). In short:

- `bash scripts/run_full_master_experiments.sh` reruns the deployed-design experiments from cached model outputs in a few minutes, without a GPU.
- `bash scripts/rerun_everything_2026-11.sh` reruns everything, including model inference on all 12,110 pictures (GPUs needed, several hours, resumable).

---

## 5. Repository layout

### In use

| Folder | What it holds |
|---|---|
| [presentation/](presentation/) | Thesis, slides, handout (sources and PDFs). `thesis_v3_updated.tex` and `advisor_document_2026-09-29.tex` stay because the checkers and rerun scripts still read them; earlier slides, meeting documents, the September source package and superseded PDFs are in `presentation/old/` (dated file names), emails to the advisors in `presentation/correspondence/` |
| `config.yaml`, `configs/`, `tier2_selection.*` | Settings and the Tier 2 picture list, read by scripts |
| `*.pt` at the root | Detector and SAM weights that some scripts load by bare file name (not in git) |
| [latex_assets/](latex_assets/) | Figures included by the thesis |
| [scripts/signal_rebuild/](scripts/signal_rebuild/) | The corrected design (Chapters 6–7, App. E and G) |
| [scripts/core_pipeline/](scripts/core_pipeline/) | The deployed design (Chapters 2–5 and 8, App. B, C and F) |
| [scripts/annotation/](scripts/annotation/), [scripts/analysis_viz/](scripts/analysis_viz/) | Label Studio export and import, plots |
| `scripts/*.py`, `scripts/*.sh` | Rerun drivers and the thesis checkers |
| `results/` | Every result file. **Not in git**: on the server only, except for the human-label files listed in REPRODUCE.md |
| [human_spatial_audit/](human_spatial_audit/) | Hand-drawn boxes: 801-picture cohort C1, 300-picture cohort C2 |
| [self_annotation_package/](self_annotation_package/) | All other hand labels (scene, object and class tasks) and the Label Studio set-up; see its README |
| `final_dataset/` | The 12,110 archive pictures and the raw model outputs (not in git) |
| `final_dataset_v1_refresh/` | Raw VLM outputs read by `extract_raw_agent_signals.py` (not in git) |
| `data/` | Detector training sets (YOLO format) |
| `runs/`, `weights/` | Trained detectors and model weights (not in git) |
| [roadmap/](roadmap/) | Thesis roadmap and the submission tracker |

### Kept for the record, not used by the thesis

[docs/history/](docs/history/) (earlier guides, run reports and progress logs),
[legacy/](legacy/) (scripts and pages formerly at the root),
`archive/` (earlier runs and superseded scripts; not in git), `reruns/` (backups taken by each full rerun),
`results_v1/`, `scratch/` (one-off checks; `calibrate_temporal.py` and `validate_temporal_agent.py`
contain synthetic data and must never be cited), `thesis_site/`, and the third-party code in
`GroundingDINO/`, `Real-ESRGAN/` and `pytorch-CycleGAN-and-pix2pix/`.

---

## 6. Ground rules

1. No script writes numbers into a `.tex` file. Results go to `results/`; the text cites them, and the checkers above hold the two together.
2. Every cited number has a script that regenerates it.
3. Hand labels cannot be regenerated. Back them up; never "clean up" their `.gitignore` exceptions.
4. An image's page id is not unique across books. Join expert labels to pictures with `scripts/core_pipeline/picture_ids.py`.
