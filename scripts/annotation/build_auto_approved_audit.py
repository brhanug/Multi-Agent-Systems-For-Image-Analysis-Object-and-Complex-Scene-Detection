#!/usr/bin/env python3
"""Sample the images the pipeline auto-approves, for a direct audit of the risk guarantee.

Why this pool and not the remaining negative controls. The blind-spot finding is already
replicated and tight: 319 confirmed negatives, 50 errors, AUROC 0.1259, CI [0.062, 0.200].
Annotating the 465 unreviewed negative controls would narrow that interval from about
0.137 to 0.104 and change no conclusion.

The claim that IS thinly supported is the operational one. The conformal guarantee at
alpha = 0.05 certifies 60.6% automatic approval at 91.1% error recall with empirical risk
0.0201. That risk is 5 errors among the 151 auto-approved images of a 249-image test
partition. The exact 95% interval on it is [0.0127, 0.0710] -- an upper bound above the
5% the guarantee certifies. Corpus-wide the rule auto-approves 8,490 images, and none
outside that 249-image partition has ever been checked.

Labelling a random sample of what the system actually auto-approves measures the
deployment risk directly. About 200 labels estimate it to roughly +/-1.9 points, enough to
separate 2% from 5%; 465 would give +/-1.3.

Outputs: self_annotation_package/task_e_auto_approved_audit/
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
CHAT = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
IMG = BASE / "final_dataset/images/restored"
PKG = BASE / "self_annotation_package/task_e_auto_approved_audit"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
LAMBDA = 0.372
SEED = 20260911


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=465)
    args = ap.parse_args()

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    ag["U_triage"] = 0.6 * ag[A4].std(axis=1, ddof=1) + 0.4 * (1 - ag[A4].mean(axis=1))
    auto = ag[ag.U_triage < LAMBDA].copy()

    already = set(pd.read_csv(SPLIT).ppn_id)
    try:
        already |= set(pd.read_csv(CHAT).ppn_id)
    except Exception:
        pass
    pool = auto[~auto.image_id.isin(already)].copy()

    rng = np.random.default_rng(SEED)
    take = min(args.n, len(pool))
    sel = pool.iloc[rng.permutation(len(pool))[:take]].copy().reset_index(drop=True)
    sel["review_order"] = np.arange(1, len(sel) + 1)
    sel["batch"] = (sel.review_order - 1) // 15 + 1
    sel["pipeline_says"] = "scene present"

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "images").mkdir(exist_ok=True)
    copied = 0
    for _, r in sel.iterrows():
        src = IMG / f"{r.image_id}.jpg"
        if src.exists():
            dst = PKG / "images" / f"{int(r.review_order):04d}_{r.image_id.replace('/', '_')}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1

    cols = ["review_order", "batch", "image_id", "U_triage", "comparison_fusion_score", "pipeline_says"]
    sel[cols].to_csv(PKG / "labeling_sheet.csv", index=False)
    meta = {
        "task": "Audit of what the pipeline auto-approves",
        "question": "Does this image actually contain a scene?",
        "why": ("The pipeline has already decided 'scene present' and auto-approved every image "
                "here. A disagreement from the rater is a false positive that would have reached "
                "the catalogue unreviewed. The rate of those IS the deployment risk the conformal "
                "guarantee bounds."),
        "auto_approved_corpus_wide": int(len(auto)),
        "eligible_unlabelled": int(len(pool)),
        "sampled": int(len(sel)),
        "images_copied": copied,
        "batches": int(sel.batch.max()),
        "lambda": LAMBDA,
        "current_evidence": {"errors": 5, "auto_approved": 151,
                             "risk_point": 0.0201, "risk_ci95": [0.0127, 0.0710],
                             "alpha": 0.05},
        "precision_gain": {"200_labels": "+/-1.94 pp", "465_labels": "+/-1.27 pp"},
    }
    (PKG / "task.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
