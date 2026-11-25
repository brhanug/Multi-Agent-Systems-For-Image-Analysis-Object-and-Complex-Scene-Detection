#!/usr/bin/env python3
"""Bring the thesis's figures to the full rerun's results, and keep them checkable afterwards.

Every entry names a figure by its place in the text and by the result it comes from. Before changing
anything, the old value is rebuilt from the pre-rerun backup and must equal what the thesis prints:
if it does not, the mapping is wrong and the entry is refused, so a mistaken source can never move a
number. Then the figure is replaced by the same key's value in the current results.

    python thesis_rerun_figures.py            # apply
    python thesis_rerun_figures.py --verify   # every figure matches the current results
    python thesis_rerun_figures.py --backup reruns/rerun_2026-11-03   # apply a later rerun
"""
import argparse, json, os, sys
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
THESIS = BASE / os.environ.get("THESIS", "presentation/thesis_v4.tex")
NOW = BASE / "results/signal_rebuild"
BACKUP = BASE / "reruns/rerun_2026-09-26/backup/results/signal_rebuild"
MARK = "⟨⟩"


def get(root, src):
    if callable(src):
        return src(root)
    if isinstance(src, tuple):
        return tuple(get(root, s) for s in src)
    f, key = src.split(":", 1)
    x = json.loads((root / f).read_text())
    for part in key.split("|"):
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


FMT = {
    "int":    lambda v: f"{int(round(v)):,}".replace(",", "{,}"),
    "f2":     lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}", "f4": lambda v: f"{v:.4f}",
    "abs2":   lambda v: f"{abs(v):.2f}", "abs3": lambda v: f"{abs(v):.3f}", "abs4": lambda v: f"{abs(v):.4f}",
    "s3":     lambda v: f"{v:+.3f}", "s4": lambda v: f"{v:+.4f}",
    "pct1":   lambda v: f"{100 * v:.1f}\\%",      # a share stored as 0..1
    "pctv1":  lambda v: f"{v:.1f}\\%",            # a value already in percent
    "pctv2":  lambda v: f"{v:.2f}\\%",
    "ci4":    lambda v: f"[{v[0]:.4f}, {v[1]:.4f}]",
    "ci3n":   lambda v: f"[{v[0]:.3f},{v[1]:.3f}]",
    "trip3":  lambda v: "/".join(f"{x:.3f}" for x in v),
    "rng3nl": lambda v: f"{v[0]:.3f}$ to\n${v[1]:+.3f}",
    "quart2": lambda v: f"{v[0]:.2f}$, ${v[1]:.2f}$, ${v[2]:.2f}$ and ${v[3]:.2f}",
    "pair3":  lambda v: f"{v[0]:.3f}/{v[1]:.3f}",
    "ci4s":   lambda v: f"[{v[0]:+.4f}, {v[1]:+.4f}]",
}

# (context with ⟨⟩ where the figure sits, "file:key|key|index", format)
E = [
    # --- redesigned signals (tab:redesigned_signals) -------------------------------------------------
    # --- repaired signals (tab:fixed_signals_character and the evaluation that follows) ---------------
    ("Measured without that restatement it scores\n  $⟨⟩$, and adding it to the spread",
     "fixed_signals.json:evaluations|0|auroc_vs_error|A_agree_fixed", "f4"),
    ("and adding it to the spread costs $⟨⟩$ ROC-AUC",
     "fixed_signals.json:evaluations|0|spread_vs_confidence|delta_fixed_spread_minus_confidence|0", "abs2"),
    ("compressed to a scalar it contributes $⟨⟩$ to trust",
     "fixed_signals.json:evaluations|0|auroc_vs_error|A_vlm_fixed", "f4"),
    # --- full Tier-1 rebuild, the corrected-channel redesign, Tier 2/3 on the full corpus -------------
    ("full Tier~1 re-run of Section~\\ref{sec:signal_rebuild}, whose rebuilt proxy takes $⟨⟩$",
     "full_tier1_rebuild.json:vs_deployed|U_SAA_distinct_rebuilt", "int"),

    # --- the scene channel (tab:scene_channel, tab:unanimity) ------------------------------------------
    # --- the corrected channel, refitted (tab:channel_refit and the passage before it) ------------------
    ("Deployed channel, hand-set $0.6/0.2/0.2$ & $42.2\\%$ & $0.6649$ & $⟨⟩$ \\\\", "corrected_channel_refit.json:configurations|0|auroc_SAA_vs_scene_presence", "f4"),
    ("Deployed channel, weights refitted & $37.8\\%$ & $⟨⟩$ &", "corrected_channel_refit.json:configurations|1|balanced_accuracy", "f4"),
    ("Deployed channel, weights refitted & $37.8\\%$ & $0.6870$ & $⟨⟩$ \\\\", "corrected_channel_refit.json:configurations|1|auroc_SAA_vs_scene_presence", "f4"),
    ("Corrected channel, hand-set $0.6/0.2/0.2$ & $39.4\\%$ & $0.6815$ & $⟨⟩$ \\\\", "corrected_channel_refit.json:configurations|2|auroc_SAA_vs_scene_presence", "f4"),
    ("& $0.6873$ & $\\mathbf{⟨⟩}$ \\\\", "corrected_channel_refit.json:configurations|3|auroc_SAA_vs_scene_presence", "f4"),
    # --- the formulation ceiling, negative evidence, the candidate that came close ---------------------
    # --- the marginal guarantee decomposed (tab:marginal_decomposed); caption rewritten 2026-09-27 ---------
    # --- why disagreement fails (the diagnosis before tab:triage_ablation) -----------------------------
    ("errors are essentially uncorrelated (pairwise $⟨⟩$), all three",
     lambda r: (lambda m: (min(m[a][b] for a in m for b in m if a < b), max(m[a][b] for a in m for b in m if a < b)))(
         json.loads((r / "why_disagreement_fails.json").read_text())["error_coupling"]["pairwise_error_correlation"]), "rng3nl"),
    ("all three wrong together on $⟨⟩$ and an effective", "why_disagreement_fails_corrected.json:error_coupling|all_three_wrong_together_pct", "pctv1"),
    ("rank of $⟨⟩$ rather than $", "why_disagreement_fails_corrected.json:redundancy|effective_rank", "f3"),
    ("rather than $⟨⟩$ --- which moves the diagnosis", "why_disagreement_fails.json:redundancy|effective_rank", "f3"),
    ("the four signals have an effective\nrank of $\\mathbf{⟨⟩}$ out of four", "why_disagreement_fails.json:redundancy|effective_rank", "f3"),
    ("its mean is $⟨⟩$ on errors and", "why_disagreement_fails.json:where_errors_sit|mean_U_on_errors", "f4"),
    ("on errors and\n$⟨⟩$ on correct predictions", "why_disagreement_fails.json:where_errors_sit|mean_U_on_correct", "f4"),
    ("on correct predictions, a difference of $⟨⟩$, while the level",
     lambda r: (lambda w: w["mean_U_on_errors"] - w["mean_U_on_correct"])(json.loads((r / "why_disagreement_fails.json").read_text())["where_errors_sit"]), "f3"),
    ("while the level differs by $⟨⟩$\n($SAA$ averages",
     lambda r: (lambda w: w["mean_SAA_on_correct"] - w["mean_SAA_on_errors"])(json.loads((r / "why_disagreement_fails.json").read_text())["where_errors_sit"]), "f3"),
    ("spread's ROC-AUC is $⟨⟩$ --- below chance",
     lambda r: tuple(v["auroc_U_within_stratum"] for v in json.loads((r / "why_disagreement_fails.json").read_text())["spread_given_level"]["within_strata"].values()), "quart2"),
    # --- what the triage can lose (tab:triage_ablation) and the archive-scale comparison ---------------
    ("Full system, 4-signal spread & $0.6625$ & $⟨⟩$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|full system (4-signal spread)|auroc_uncertainty", "f4"),
    ("Full system, 4-signal spread & $0.6625$ & $0.5513$ & $⟨⟩$ \\\\", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|full system (4-signal spread)|auroc_confidence", "f4"),
    ("Remove $A^{\\text{agree}}$ & $0.6625$ & $⟨⟩$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove A_agree|auroc_uncertainty", "f4"),
    ("Remove $A^{\\text{agree}}$ & $0.6625$ & $0.5322$ & $⟨⟩$ \\\\", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove A_agree|auroc_confidence", "f4"),
    ("\\textbf{Remove the spread entirely} & $0.6625$ & $\\mathbf{⟨⟩}$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the spread entirely (confidence only)|auroc_uncertainty", "f4"),
    ("\\textbf{Remove the spread entirely} & $0.6625$ & $\\mathbf{0.8309}$ & $⟨⟩$ \\\\", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the spread entirely (confidence only)|auroc_confidence", "f4"),
    ("Remove the detector & $\\mathbf{⟨⟩}$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the detector|balanced_accuracy", "f4"),
    ("Remove the detector & $\\mathbf{0.5521}$ & $⟨⟩$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the detector|auroc_uncertainty", "f4"),
    ("Remove the detector & $\\mathbf{0.5521}$ & $0.3884$ & $⟨⟩$ \\\\", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the detector|auroc_confidence", "f4"),
    ("Remove the scene classifier & $0.6672$ & $⟨⟩$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the scene classifier|auroc_uncertainty", "f4"),
    ("Remove the scene classifier & $0.6672$ & $0.4787$ & $⟨⟩$ \\\\", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the scene classifier|auroc_confidence", "f4"),
    ("Remove the VLM & $0.6483$ & $⟨⟩$ &", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the VLM|auroc_uncertainty", "f4"),
    ("raises the triage score's discrimination from $⟨⟩$ to", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|full system (4-signal spread)|auroc_uncertainty", "f4"),
    ("discrimination from $0.5513$ to\n$⟨⟩$. Removing the scene", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the spread entirely (confidence only)|auroc_uncertainty", "f4"),
    ("collapses the system to\nnear chance ($⟨⟩$): it is the one", "redesigned_corpus_and_recommendation.json:ablation_what_can_go|remove the detector|balanced_accuracy", "f4"),
    ("on the spread (as designed) & $0.370$ & $⟨⟩$ \\ ($", "redesigned_corpus_and_recommendation.json:corpus_scale|triage on spread (as designed)|corpus_auto_approved", "int"),
    ("& $1{,}196$ \\ ($9.9\\%$) & $⟨⟩$ & satisfied", "redesigned_corpus_and_recommendation.json:corpus_scale|triage on spread (as designed)|error_recall_pct", "pctv1"),
    ("on confidence & $0.368$ & $\\mathbf{⟨⟩}$ \\ $(", "redesigned_corpus_and_recommendation.json:corpus_scale|triage on confidence|corpus_auto_approved", "int"),
    ("archive automatically --- $⟨⟩$ images against", "redesigned_corpus_and_recommendation.json:corpus_scale|triage on confidence|corpus_auto_approved", "int"),
    ("images against $⟨⟩$ --- at the same risk", "redesigned_corpus_and_recommendation.json:corpus_scale|triage on spread (as designed)|corpus_auto_approved", "int"),
    # added 5 Nov 2026: cells fixed by hand after the 3 Nov rerun
]


def main():
    global BACKUP
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--backup", default=None, help="a rerun folder whose backup/results/signal_rebuild holds the old values")
    a = ap.parse_args()
    if a.backup:
        BACKUP = Path(a.backup) / "backup/results/signal_rebuild"
    t = THESIS.read_text()
    changed = skipped = 0; problems = []
    for ctx, src, fmt in E:
        f = FMT[fmt]
        vn, vo = get(NOW, src), get(BACKUP, src)
        new = f(vn)
        # the thesis may have rounded a half the other way (0.2925 printed as 0.293): accept either
        nudge = (lambda d: tuple(x + d for x in vo)) if isinstance(vo, tuple) else (lambda d: vo + d)
        olds = {f(vo), f(nudge(1e-7)), f(nudge(-1e-7))}
        new_ctx = ctx.replace(MARK, new)
        old = next((o for o in olds if t.count(ctx.replace(MARK, o)) == 1), f(vo))
        old_ctx = ctx.replace(MARK, old)
        if a.verify:
            if t.count(new_ctx) != 1: problems.append(f"not found once: {new_ctx[:90]!r}")
            continue
        if old == new:
            skipped += 1
            if t.count(new_ctx) != 1: problems.append(f"unchanged value but context not found: {old_ctx[:90]!r}")
            continue
        if t.count(new_ctx) == 1 and t.count(old_ctx) == 0:
            skipped += 1; continue                                  # applied on an earlier run
        if t.count(old_ctx) != 1:
            problems.append(f"old context found {t.count(old_ctx)}x (mapping wrong?): {old_ctx[:90]!r}"); continue
        t = t.replace(old_ctx, new_ctx); changed += 1
    if not a.verify and not problems:
        THESIS.write_text(t)
    for p in problems: print("  PROBLEM", p)
    if a.verify:
        print(f"{len(E) - len(problems)} of {len(E)} thesis figures match the current results"); return 1 if problems else 0
    print(f"changed {changed}, already current {skipped}, problems {len(problems)}" + ("" if not problems else " -- nothing written"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
