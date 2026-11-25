#!/usr/bin/env python3
"""Check every headline figure in the advisor document against the result file it came from.

Each row below names a number as the document prints it and where that number is computed. The check
fails if the document no longer contains the current value, so a result that moves cannot quietly
leave a stale figure behind in the text sent to an advisor.

    python check_advisor_document.py
"""
import json
import re
import sys
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
DOC = BASE / "presentation/advisor_document_2026-09-29.tex"
R = BASE / "results/signal_rebuild"


def j(p):
    return json.loads((R / p).read_text())


def main():
    text = DOC.read_text()
    b1 = j("b1/b1_per_system.json"); votes = j("class_scores/teacher_label_votes.json")
    c4 = j("c4/c4_preregistered.json"); base = j("baselines/baselines.json")
    tv = j("b3/teacher_vs_student.json"); cand = j("student_rebuild/teacher_candidates_300.json")
    kap = json.loads((BASE / "results/multi_agent/per_class_human_kappa.json").read_text())
    ps = j("pseudo_label_statistics.json")
    fd = {e["class"]: e for e in votes["sets"]["final_dataset"]["per_class"]}
    v1 = {e["class"]: e for e in votes["sets"]["v1_refresh"]["per_class"]}
    coco = {e["class"]: e for e in base["detection_baseline_kim_im_mandl_starting_point"]["per_class"]}
    ct = lambda k, c: cand["results"][k][c]["mean_ap50"]

    checks = [
        ("B1 mean F1, LLaVA", f"{b1['macro']['llava']['shared8_cv_f1']:.2f}"),
        ("B1 mean F1, fusion", f"{b1['macro']['fusion']['shared8_cv_f1']:.2f}"),
        ("B1 mean F1, SigLIP", f"{b1['macro']['siglip']['shared8_cv_f1']:.2f}"),
        ("B1 mean F1, detector", f"{b1['macro']['detector']['shared8_cv_f1']:.2f}"),
        ("B1 always-yes baseline", f"{b1['macro']['always_yes_baseline']['shared8_cv_f1']:.2f}"),
        ("B2 person, both agree, final_dataset", f"{fd['person']['share_all_independent_systems_agree']:.0%}".replace("%", "\\%")),
        ("B2 child, both agree, final_dataset", f"{fd['child']['share_all_independent_systems_agree']:.0%}".replace("%", "\\%")),
        ("B2 building, both agree, v1_refresh", f"{v1['building']['share_all_independent_systems_agree']:.0%}".replace("%", "\\%")),
        ("C4 per-class disagreement on the 300", f"{c4['set_A_300']['auroc']['per-class disagreement']:.3f}"),
        ("C4 plain confidence on the 300", f"{c4['set_A_300']['auroc']['plain confidence']:.3f}"),
        ("C4 registered difference", f"{abs(c4['set_A_300']['tested_difference']['per-class disagreement minus plain confidence']):.3f}"),
        ("kappa, landscape", f"{[r for r in kap['per_category'] if r['category'] == 'landscape'][0]['cohens_kappa']:.2f}"),
        ("kappa, teaching", f"{abs([r for r in kap['per_category'] if r['category'] == 'teaching'][0]['cohens_kappa']):.2f}"),
        ("baseline, COCO detector on person", f"{coco['person']['auroc']:.2f}"),
        ("baseline, COCO detector on vehicle", f"{coco['vehicle']['auroc']:.2f}"),
        ("teacher failures among the misses", str(tv["misses_on_the_300"]["teacher_failure"])),
        ("training failures among the misses", str(tv["misses_on_the_300"]["training_failure"])),
        ("pictures with a teacher label", str(tv["pictures_of_the_300_with_teacher_labels"])),
        ("student finds nothing", f"{tv['archive']['student_finds_nothing']:,}"),
        ("pseudo-label boxes", f"{ps['boxes']:,}" if "boxes" in ps else f"{ps.get('total_boxes', 0):,}"),
        ("rebuilt student, mean AP50", f"{ct('all_300', 'new_student'):.3f}"),
        ("deployed student, mean AP50", f"{ct('all_300', 'student'):.3f}"),
        ("GroundingDINO, mean AP50", f"{ct('all_300', 'gdino'):.3f}"),
        ("rebuilt student on the 164", f"{ct('164_not_in_training', 'new_student'):.3f}"),
    ]

    # the corrected Tier-1 section
    t1 = j("corrected_tier1_end_to_end.json"); red = j("agent_redundancy.json"); sc = j("scene_channel_fix.json")
    v = t1["variants"]
    checks += [
        ("Tier 1, deployed scene AUROC", f"{v['deployed']['scene_ranking_auroc']:.3f}"),
        ("Tier 1, corrected scene AUROC", f"{v['corrected (rebuilt detector)']['scene_ranking_auroc']:.3f}"),
        ("Tier 1, deployed test error", f"{v['deployed']['test_error_rate']:.3f}"),
        ("Tier 1, corrected test error", f"{v['corrected (rebuilt detector)']['test_error_rate']:.3f}"),
        ("Tier 1, corrected with old detector", f"{v['corrected (deployed detector)']['scene_ranking_auroc']:.3f}"),
        ("object agent alone", f"{t1['agents']['object']['auroc_on_test_alone']:.3f}"),
        ("vlm agent alone", f"{t1['agents']['vlm']['auroc_on_test_alone']:.3f}"),
        ("scene agent alone", f"{t1['agents']['scene']['auroc_on_test_alone']:.3f}"),
        ("scene softmax AUROC", f"{sc['which_channel_measures_scene_presence']['auroc_softmax']:.3f}"),
        ("scene sigmoid AUROC", f"{sc['which_channel_measures_scene_presence']['auroc_sigmoid']:.3f}"),
        ("softmax corpus minimum", f"{sc['the_defect']['softmax_min']:.4f}"),
        ("joint states of the deployed vector",
         str(red["the_deployed_uncertainty_signal"]["distinct_values_over_12110_pictures"])),
    ]

    # the absence router and the confidence-scored triage
    checks.append(("absence router, scene AUROC",
                   f"{v['corrected + absence router']['scene_ranking_auroc']:.3f}"))
    for tag, label in (("deployed", "deployed"), ("corrected (rebuilt detector)", "corrected"),
                       ("corrected + absence router", "absence")):
        for c in v[tag]["conformal_risk_control_on_confidence"]:
            if c["alpha"] in (0.05, 0.10, 0.15):
                checks.append((f"confidence triage, {label} at {c['alpha']}",
                               f"{c['auto_approval_pct']:.1f}\\%"))

    # figures a full rerun moved that were not yet held to their source (added 2026-09-27)
    pil = j("class_scores/llava_class_prompt_pilot_summary.json")["macro"]
    llb = b1["macro"]["llava"]
    ca = j("class_agreement/class_agreement.json")["c3_on_300_vs_human"]["text"]
    na = cand["results"]["all_300"]["new_student"]["per_class_ap50"]
    absr = v["corrected + absence router"]; reb = v["corrected (rebuilt detector)"]
    checks += [
        ("LLaVA phrasings agree", f"{100 * pil['A1_A2_same_decision']:.1f}\\%"),
        ("LLaVA F1, blurry or faded", f"LLaVA {llb['shared_cv_f1_blurry_or_faded']:.2f} on the blurry"),
        ("LLaVA F1, clear", f"{llb['shared_cv_f1_clear']:.2f} on the clear"),
        ("C4, disagreement on the 164", f"{c4['set_A_164_not_in_detector_training']['auroc']['per-class disagreement']:.3f}"),
        ("C4, global disagreement on the test split", f"{c4['set_B_test_partition']['auroc']['global disagreement']:.3f}"),
        ("text class, SigLIP-LLaVA vs LLaVA-rater", f"{ca['machine_pair_kappa']['siglip-llava']:.3f} against {ca['system_vs_human_kappa']['llava']:.3f}"),
        ("rebuilt student, building AP50", f"building 0.05 to {na['building']:.2f}"),
        ("rebuilt student, vehicle AP50", f"0 to\n{na['vehicle']:.2f}"),
        ("corrected, confidence at errors", f"${reb['plain_confidence_auroc']:.3f}$ against disagreement's ${reb['uncertainty_auroc_at_predicting_errors']:.3f}$"),
        ("absence router, test error", f"{absr['test_error_rate']:.3f}"),
    ]

    # the best configuration over 200 redraws
    bc = j("best_config_repeated.json")
    for v in ("deployed", "corrected", "corrected + absence"):
        for m in ("scene_auroc", "test_error"):
            s = bc["per_pipeline"][v][m]
            checks.append((f"redraws, {v} {m}", f"{s['mean']:.3f} \\,[{s['p2.5']:.3f}, {s['p97.5']:.3f}]"))
    pdiff = bc["paired_differences"]
    for al in ("0.05", "0.1", "0.15"):
        s = pdiff[f"auto-approval at {al}, corrected + absence (confidence) minus deployed (disagreement)"]
        checks.append((f"redraws, approval gain at {al}", f"{100 * s['mean']:.1f}"))
        checks.append((f"redraws, approval gain low at {al}", f"{100 * s['p2.5']:.1f}"))
        checks.append((f"redraws, approval gain high at {al}", f"{100 * s['p97.5']:.1f}"))
    for k, key in (("ranking gain share", "scene AUROC, corrected + absence minus deployed"),
                   ("error gain share", "test error, deployed minus corrected + absence"),
                   ("absence gain share", "scene AUROC, absence router's own gain"),
                   ("triage share", "auto-approval at 0.05, confidence minus disagreement triage (corrected + absence)")):
        checks.append((f"redraws, {k}", f"{100 * pdiff[key]['share_of_draws_above_zero']:.1f}\\%"))

    # the out-of-pool check and the label agreement behind it
    op = j("out_of_pool_check.json"); ae = op["all_eligible"]; la = op["label_agreement_on_shared_pictures"]
    checks += [
        ("out of pool, deployed AUROC", f"{ae['pipelines']['deployed (disagreement triage)']['scene_auroc']:.3f}"),
        ("out of pool, corrected AUROC", f"{ae['pipelines']['corrected + absence (confidence triage)']['scene_auroc']:.3f}"),
        ("out of pool, deployed error", f"{ae['pipelines']['deployed (disagreement triage)']['error_rate']:.3f}"),
        ("out of pool, corrected error", f"{ae['pipelines']['corrected + absence (confidence triage)']['error_rate']:.3f}"),
        ("out of pool, AUROC difference", f"{ae['paired']['scene AUROC, corrected minus deployed']['mean']:.3f}"),
        ("out of pool, pictures", str(ae["pictures"])),
        ("label agreement", f"{100 * la['agreement']:.1f}\\%"),
        ("label kappa", f"{la['cohens_kappa']:.2f}"),
        ("rater stricter", str(la["rater_no_scene_but_pool_scene"])),
    ]

    # the label-shift check behind the out-of-pool reading (added 2026-09-29)
    ls = j("label_shift_check.json")
    PL = ls["all_197"]["scene pool's labels"]; RL = ls["all_197"]["second rater's labels"]; AG = ls["where_the_labels_agree"]
    checks += [
        ("label shift, pool labels", f"${PL['corrected_auroc']:.3f}$ against ${PL['deployed_auroc']:.3f}$"),
        ("label shift, rater labels", f"${RL['deployed_auroc']:.3f}$\nagainst ${RL['corrected_auroc']:.3f}$"),
        ("label shift, labels agree", f"${AG['corrected_auroc']:.3f}$ against\n${AG['deployed_auroc']:.3f}$"),
        ("label shift, pictures", str(ls["pictures_with_both_labels"])),
    ]

    # the effective-rank gate
    er = j("effective_rank_gate.json")
    checks += [("effective rank, deployed three", f"{er['deployed']['three measured signals']:.2f}"),
               ("effective rank, corrected", f"{er['corrected']['corrected']['three calibrated P(scene)']:.2f}"),
               ("effective rank, corrected + absence",
                f"{er['corrected']['corrected + absence']['three calibrated P(scene)']:.2f}"),
               ("object/vlm correlation", f"{er['corrected']['corrected']['pairwise correlation']['object/vlm']:.3f}")]

    # the repeated-split risk check
    rc = j("corrected_crc_repeated.json")["results"]
    for al in ("0.01", "0.05", "0.1", "0.15"):
        checks.append((f"repeated CRC, corrected mean risk at {al}",
                       f"{rc['corrected'][al]['mean_realised_risk']:.4f}"))
        checks.append((f"repeated CRC, deployed mean risk at {al}",
                       f"{rc['deployed'][al]['mean_realised_risk']:.4f}"))
    for who, al in (("corrected", "0.1"), ("deployed", "0.1"), ("corrected", "0.15"), ("deployed", "0.15")):
        checks.append((f"repeated CRC, {who} auto-approval at {al}",
                       f"{100 * rc[who][al]['mean_auto_approval']:.1f}\\%"))
    bad = []
    for name, value in checks:
        if value not in text:
            bad.append((name, value))
    for name, value in checks:
        mark = "MISSING" if (name, value) in bad else "ok"
        print(f"  {mark:8s} {name:42s} {value}")
    if bad:
        print(f"\n{len(bad)} figure(s) in the document no longer match the results")
        return 1
    print(f"\nall {len(checks)} checked figures match the results they come from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
