#!/bin/bash
# Rerun the whole thesis pipeline from the pictures up: every model's inference, the student detector
# retrained, the core master pipeline, every signal-rebuild analysis, both documents, and a report of
# what changed against a backup taken first.
#
# 3 November 2026 version of rerun_everything.sh: new run folder; adds the core scripts fixed for the
# book-and-page join, retrains the six label-filter students, and ends with the three thesis checks.
#
# Detached and resumable. Each step leaves a marker in $RUN/state; starting the script again skips
# finished steps, so a crash or a closed session loses nothing. A failed step is logged and marked,
# and the steps that do not depend on it carry on.
#
#   setsid nohup bash scripts/rerun_everything.sh > /dev/null 2>&1 &
#   tail -f reruns/rerun_2026-11-03/rerun.log
set -u
BASE=/data/brhanu/thesis_project
PY=$BASE/.venv/bin/python3
SR=$BASE/results/signal_rebuild
RUN=$BASE/reruns/rerun_2026-11-03
BK=$RUN/backup
LOG=$RUN/rerun.log
ST=$RUN/state
mkdir -p "$ST" "$RUN/logs"
cd "$BASE"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_() { [ -f "$ST/$1.done" ]; }
# step <name> <command...>: run once, record the outcome, never stop the whole run
step() {
  local name=$1; shift
  if done_ "$name"; then return 0; fi
  say "START $name"
  local t0=$(date +%s)
  if "$@" >> "$RUN/logs/$name.log" 2>&1; then
    touch "$ST/$name.done"; rm -f "$ST/$name.failed"
    say "done  $name ($(( ($(date +%s) - t0) / 60 )) min)"
  else
    local rc=$?; echo "$rc" > "$ST/$name.failed"
    say "FAIL  $name (exit $rc) -- see logs/$name.log"
  fi
}
shards() {  # shards <script> <gpu list...>: one shard per GPU, all in parallel
  local script=$1; shift; local n=$#; local i=0; local pids=()
  for g in "$@"; do
    CUDA_VISIBLE_DEVICES=$g $PY "scripts/signal_rebuild/$script" --shard $i --num-shards $n &
    pids+=($!); i=$((i + 1))
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}

say "=== FULL RERUN (re)started, pid $$ ==="

# ---- 0. back up, then clear every model output so nothing is reused ----------------------------
backup() {
  mkdir -p "$BK"
  cp -a results "$BK/results"
  [ -d runs/detect/yolo11_gdino_consensus ] && mv runs/detect/yolo11_gdino_consensus "$BK/run_yolo11_gdino_consensus"
  [ -d data/yolo11_gdino_consensus ] && cp -a data/yolo11_gdino_consensus "$BK/data_yolo11_gdino_consensus" \
      && rm -rf data/yolo11_gdino_consensus
  cp -a presentation/advisor_document_2026-09-29.pdf presentation/thesis_v3_updated.pdf "$BK/" 2>/dev/null
  cp -a latex_assets "$BK/latex_assets"
  mkdir -p "$BK/label_filter_runs"
  for d in runs/detect/label_filter_*; do [ -d "$d" ] && mv "$d" "$BK/label_filter_runs/"; done
  rm -f "$SR"/label_filter/detections_*.json
  rm -rf "$SR/llava_full" "$SR/kosmos_full" "$SR/class_scores/llava_class_scores" "$SR/student_rebuild/gdino_labels"
  rm -f "$SR/siglip_scene_full.jsonl" "$SR/yolo_full_corpus.jsonl" "$SR/siglip_embeddings.npz" \
        "$SR/negative_evidence_scores.npz" "$SR/class_scores/siglip_class_scores.npz" \
        "$SR/class_scores/yolo_class_scores.jsonl" "$SR/class_scores/new_student_class_scores.jsonl" \
        "$SR/class_scores/new_student_class_scores_conf0001.jsonl" "$SR/class_scores/llava_class_prompt_pilot.jsonl" \
        "$SR/student_rebuild/detections_300.json" "$SR/student_rebuild/detections_300_new_student.json" \
        "$SR/baselines/coco_detector_scores_300.json" "$SR/kosmos_generation_pilot_outputs.jsonl"
}
step 00_backup_and_clear backup

# ---- 1-5. inference on GPUs 0 and 1 (GPUs 2 and 3 are left to the other users of this machine) --------------------------------------------------------------
step 01_llava_scene_answers  shards run_llava_full_corpus.py 0 1
step 02_llava_class_scores   shards run_llava_class_scores.py 0 1
# ---- 3. in parallel: LLaVA prompt pilot | SigLIP and the deployed detector | GroundingDINO teacher ----
small_models() {  # all on GPU 1; the scene scorer must precede the negative-evidence detector, which reads it
  CUDA_VISIBLE_DEVICES=1 $PY scripts/signal_rebuild/run_agent4_embeddings_full_corpus.py --embed || return 1
  CUDA_VISIBLE_DEVICES=1 $PY scripts/signal_rebuild/run_siglip_class_scores.py || return 1
  CUDA_VISIBLE_DEVICES=1 $PY scripts/signal_rebuild/run_siglip_scene_full_corpus.py || return 1
  CUDA_VISIBLE_DEVICES=1 $PY scripts/signal_rebuild/run_negative_evidence_detector.py || return 1
  $PY scripts/signal_rebuild/run_yolo_full_corpus.py --device 1 --batch 48 || return 1
  $PY scripts/signal_rebuild/run_yolo_class_scores.py --device 1 || return 1
}
( step 03a_llava_prompt_pilot env CUDA_VISIBLE_DEVICES=0 $PY scripts/signal_rebuild/run_llava_class_prompt_pilot.py ) &
( step 03b_siglip_and_deployed_detector small_models ) &
wait
step 03c_gdino_teacher_labels shards run_gdino_teacher_labels.py 0 1

# ---- 4-5. the per-class matrix, which the student dataset reads for LLaVA's confirmation --------------
step 04_class_score_matrix $PY scripts/signal_rebuild/build_class_score_matrix.py
step 05_student_dataset    $PY scripts/signal_rebuild/build_gdino_student_dataset.py

# ---- 6. in parallel: retrain the student on GPU 0 | Kosmos on GPU 1 --------------------------------
train_student() {
  local run=$BASE/runs/detect/yolo11_gdino_consensus
  finished() { [ -f "$run/weights/last.pt" ] && $PY -c "import torch,sys; ck=torch.load('$run/weights/last.pt',map_location='cpu',weights_only=False); sys.exit(0 if ck.get('epoch',0)==-1 else 1)"; }
  finished && return 0
  if [ -f "$run/weights/last.pt" ]; then
    $PY -c "from ultralytics import YOLO; YOLO('$run/weights/last.pt').train(resume=True)"
  else
    $PY scripts/signal_rebuild/train_gdino_student.py --device 0
  fi
  local tries=0
  until finished; do
    tries=$((tries + 1)); [ $tries -gt 5 ] && return 1
    $PY -c "from ultralytics import YOLO; YOLO('$run/weights/last.pt').train(resume=True)"
  done
}
( step 06a_train_student train_student ) &
( step 06b_kosmos shards run_kosmos_full_corpus.py 1 ) &
wait

# ---- 7. everything that needs the new student --------------------------------------------------------
W=runs/detect/yolo11_gdino_consensus/weights/best.pt
step 07a_student_scores_001  $PY scripts/signal_rebuild/run_yolo_class_scores.py --device 0 --weights $W \
     --out $SR/class_scores/new_student_class_scores.jsonl
step 07b_student_scores_0001 $PY scripts/signal_rebuild/run_yolo_class_scores.py --device 0 --conf 0.001 --weights $W \
     --out $SR/class_scores/new_student_class_scores_conf0001.jsonl
step 07c_teacher_candidates  env CUDA_VISIBLE_DEVICES=0 $PY scripts/signal_rebuild/run_teacher_candidates_on_300.py
step 07d_kosmos_pilot        $PY scripts/signal_rebuild/run_kosmos_generation_pilot.py --device cuda:1

# ---- 8. the core master pipeline (17 steps) ---------------------------------------------------------
step 08_core_master bash scripts/run_full_master_experiments.sh
step 08b_per_class_human_kappa $PY scripts/core_pipeline/run_per_class_human_kappa.py
for f in exp20_error_targeted_awlf exp21_full_agent_leaveoneout exp30_shapley_weighted_fusion \
         run_real_expert_rq2_experiment run_expert_calibration_plot run_formulation_sensitivity \
         compute_colibri_c1_baseline_for_europeana; do
  step "08c_$f" $PY "scripts/core_pipeline/$f.py"
done
step 08d_awlf_figure cp results/figures/exp20_error_awlf_roc_curve.png latex_assets/exp20_error_awlf_roc_curve.png

# ---- 8e. label filter: rebuild the three training sets, retrain six students on GPUs 0-1, score, summarise
step 08e_label_filter_datasets $PY scripts/signal_rebuild/build_label_filter_datasets.py
label_filter_pair() {  # label_filter_pair <variant> <gpu>: both seeds, one after the other
  local v=$1 g=$2 seed
  for seed in 20260919 20261007; do
    (cd scripts/signal_rebuild && $PY run_label_filter_experiment.py train --variant "$v" --seed "$seed" --device "$g") || return 1
    (cd scripts/signal_rebuild && CUDA_VISIBLE_DEVICES=$g $PY run_label_filter_experiment.py score --variant "$v" --seed "$seed" --device 0) || return 1
  done
}
( step 08f_label_filter_all       label_filter_pair all 0 ) &
( step 08f_label_filter_random    label_filter_pair random 1 ) &
wait
step 08f_label_filter_consensus label_filter_pair consensus 0
step 08g_label_filter_summary bash -c "cd scripts/signal_rebuild && $PY run_label_filter_experiment.py summary"

# ---- 9. every signal-rebuild analysis, in the order it was built ------------------------------------
SKIP="run_siglip_scene_full_corpus run_yolo_full_corpus run_kosmos_full_corpus run_llava_full_corpus
run_agent4_embeddings_full_corpus run_llava_class_scores run_llava_class_prompt_pilot run_siglip_class_scores
run_yolo_class_scores run_gdino_teacher_labels build_gdino_student_dataset train_gdino_student
run_teacher_candidates_on_300 run_kosmos_generation_pilot run_negative_evidence_detector tier1_signals
check_advisor_document check_agent_readouts
build_label_filter_datasets run_label_filter_experiment"
step 09_000_agent4_linking env CUDA_VISIBLE_DEVICES=0 $PY scripts/signal_rebuild/run_agent4_embeddings_full_corpus.py --link
n=0
for f in $(git log --reverse --diff-filter=A --name-only --format= -- 'scripts/signal_rebuild/*.py' \
           | sed 's#scripts/signal_rebuild/##; s#\.py$##' | awk '!seen[$0]++'); do
  case " $(echo $SKIP) " in *" $f "*) continue ;; esac
  [ -f "scripts/signal_rebuild/$f.py" ] || continue
  n=$((n + 1))
  step "09_$(printf %03d $n)_$f" env CUDA_VISIBLE_DEVICES=0 $PY "scripts/signal_rebuild/$f.py"
done

# ---- 9b. the corrected-channel variants, which five scripts write only when asked for them -----------
# (found missing after the first full rerun: run without arguments, these scripts write only their
#  softmax-channel result, so the _corrected.json files the thesis cites were left stale)
for f in rebuild_from_full_tier1 run_redesigned_signals run_redesigned_full_corpus_and_recommend \
         diagnose_why_disagreement_fails run_fixed_signals; do
  step "09b_${f}_sigmoid" $PY "scripts/signal_rebuild/$f.py" --channel sigmoid
done

# ---- 10. what changed, the documents, and every check ------------------------------------------------
step 10a_change_report $PY scripts/rerun_change_report.py "$BK" "$RUN/change_report.md"
step 10b_advisor_pdf   bash -c "mkdir -p $RUN/pdf && cd presentation && ~/.local/bin/tectonic -X compile advisor_document_2026-09-29.tex --outdir $RUN/pdf --keep-logs"
step 10c_thesis_pdf    bash -c "mkdir -p $RUN/pdf && cd presentation && ~/.local/bin/tectonic -X compile thesis_v4.tex --outdir $RUN/pdf --keep-logs"
step 10d_tier1_guards  $PY scripts/signal_rebuild/tier1_signals.py
step 10e_readouts      $PY scripts/signal_rebuild/check_agent_readouts.py
step 10f_document_figures $PY scripts/signal_rebuild/check_advisor_document.py

step 10g_check_c1_figures      $PY scripts/thesis_c1_figures_check.py
step 10h_check_chapter6        $PY scripts/thesis_v4_chapter6_check.py
step 10i_check_thesis_figures  env THESIS=presentation/thesis_v4.tex $PY scripts/thesis_rerun_figures.py --verify

say "=== FULL RERUN FINISHED: $(ls $ST/*.done 2>/dev/null | wc -l) steps done, $(ls $ST/*.failed 2>/dev/null | wc -l) failed ==="
touch "$RUN/FINISHED"
