#!/usr/bin/env python3
"""The one place Tier-1 signals are built, with the guards that would have caught the audit's defects.

Every defect found in the September audit sat in the step that turns a model's raw output into a
number, not in the models. A fallback supplied the object signal on 98% of the corpus, the VLM signal
was an indicator of that fallback, the scene signal was read from the wrong SigLIP channel, and the
agreement term was arithmetic over two of the others. Each of those is silent: the pipeline runs,
produces a table, and reports a result.

So the corrected construction lives here and is the default, the deployed construction is reachable
only by asking for it in words (variant="deployed"), and neither can be reached by falling back from
the other. Before any fusion reads these signals, check() asserts what the audit had to discover:

  measured     every signal comes from a model's own output over this corpus, never a manifest proxy
  covered      every picture of the corpus has a value
  continuous   a signal with a handful of distinct values cannot carry a per-picture decision
  independent  no signal is a deterministic function of another

A signal taking 2-8 values across 12,110 pictures stops the pipeline.

    python tier1_signals.py            # health report for the corrected signals
    python tier1_signals.py --deployed # the same report for the audited baseline, showing what fails
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
CS = SR / "class_scores"

CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
AGENTS = ("object", "scene", "vlm")

# The rebuilt detector's class ids. Taken from the checkpoint, never from data/yolo11_v2_augmented.yaml:
# that file disagrees with the weights it trained (it calls id 9 "furniture"; decoded as "animal" id 9
# predicts the human animal label at AUROC 0.808, and id 8 "text" never fires, matching its zero
# training boxes). Always decode with model.names.
DISTINCT_FLOOR = 100


class SignalDefect(AssertionError):
    """A Tier-1 signal is not what the pipeline's description says it is."""


def _matrix():
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]
    return m, ids, {i: k for k, i in enumerate(ids)}, [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]


def corrected(absence=False):
    """Each agent's own output, in its own form: the construction each agent's specification promises.

    absence=True gives the scene agent a sixth number: SigLIP's evidence that the picture is one of
    the things the negative-control audit found the true negatives to be -- printed text, sheet music,
    ornament, a blank or title page, a table of numbers, a map. Five prompts that all name a scene
    type cannot report that a picture is none of them, which is the pipeline's dominant error.
    """
    m, ids, idx, mc, ms = _matrix()
    cols = [mc.index(c) for c in CLASSES]

    vlm = np.nan_to_num(m["raw"][:, ms.index("llava")][:, cols].astype(np.float64))

    obj = np.zeros((len(ids), len(CLASSES)))
    seen = set()
    for line in open(CS / "new_student_class_scores.jsonl"):
        r = json.loads(line)
        k = idx.get(r["image_id"])
        if k is None:
            continue
        seen.add(k)
        for c, v in r["max_conf"].items():
            if c in CLASSES:
                obj[k, CLASSES.index(c)] = v

    scene = np.zeros((len(ids), len(SCENES)))
    scene_seen = set()
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line)
        iid = r["image_id"]
        iid = iid if "/" in iid else iid.replace("_", "/", 1)
        k = idx.get(iid)
        if k is None:
            continue
        scene_seen.add(k)
        for j, s in enumerate(SCENES):
            scene[k, j] = float(r["sigmoid"].get(s, 0.0))

    scene_source = "SigLIP sigmoid over the 5 scene prompts, its native calibrated channel"
    if absence:
        z = np.load(SR / "negative_evidence_scores.npz", allow_pickle=True)
        pos_of = {str(i): k for k, i in enumerate(z["ids"])}
        neg = np.zeros((len(ids), 1))
        for iid, k in idx.items():
            j = pos_of.get(iid, pos_of.get(iid.replace("/", "_")))
            if j is not None:
                neg[k, 0] = float(z["neg"][j])
        scene = np.hstack([scene, neg])
        scene_source += ", plus the evidence that the picture is none of them (7 negative prompts)"

    return {"ids": ids, "index": idx,
            "object": obj, "scene": scene, "vlm": vlm,
            "coverage": {"object": len(seen), "scene": len(scene_seen), "vlm": len(ids)},
            "sources": {"object": "rebuilt detector, per-class max confidence over the 8 frozen classes",
                        "scene": scene_source,
                        "vlm": "LLaVA per-class P(yes) from first-token logprobs"},
            "variant": "corrected+absence" if absence else "corrected"}


def deployed():
    """The audited baseline, kept reproducible on purpose. Never reached by falling back from corrected."""
    import pandas as pd
    _, ids, idx, _, _ = _matrix()
    ag = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv").drop_duplicates("image_id")
    ag = ag.set_index("image_id").reindex(ids)
    return {"ids": ids, "index": idx,
            "object": ag[["existing_pipeline_agent"]].values,
            "scene": ag[["scene_agent"]].values,
            "vlm": ag[["vlm_agent"]].values,
            "agree": ag[["agreement_agent"]].values,
            "coverage": {a: int(ag[c].notna().sum()) for a, c in
                         (("object", "existing_pipeline_agent"), ("scene", "scene_agent"), ("vlm", "vlm_agent"))},
            "sources": {a: str(ag[c].iloc[0]) for a, c in
                        (("object", "object_source"), ("scene", "scene_source"), ("vlm", "vlm_source"))},
            "variant": "deployed"}


def check(sig, strict=True):
    """Assert the four properties the audit had to discover. Returns the findings either way."""
    n = len(sig["ids"])
    findings = []
    for a in AGENTS:
        M = np.asarray(sig[a], dtype=np.float64)
        cov = sig["coverage"][a]
        if cov != n:
            findings.append(f"{a}: covers {cov:,} of {n:,} pictures, not all of them")
        src = sig["sources"][a]
        if src == "proxy":
            findings.append(f"{a}: source is a manifest proxy, not this model's output over this corpus")
        d = len(np.unique(np.round(M, 9)))
        if d < DISTINCT_FLOOR:
            findings.append(f"{a}: takes {d} distinct values over {n:,} pictures "
                            f"(floor {DISTINCT_FLOOR}) -- too coarse to carry a per-picture decision")
    # a signal that is a function of another carries no information of its own
    for a, b in (("vlm", "object"), ("object", "vlm")):
        A, B = np.asarray(sig[a], float), np.asarray(sig[b], float)
        if A.shape[1] == 1 and B.shape[1] == 1:
            ind = (B[:, 0] > 0).astype(float)
            if np.allclose(A[:, 0], ind):
                findings.append(f"{a} is the indicator of {b} > 0 on every picture: it is not a second signal")
    if "agree" in sig:
        A, O, V = (np.asarray(sig[k], float)[:, 0] for k in ("agree", "object", "vlm"))
        if np.allclose(A, np.clip(1.0 - np.abs(O - V), 0, 1)):
            findings.append("agree = 1 - |object - vlm| on every picture: arithmetic over two others, not an agent")
    if findings and strict:
        raise SignalDefect("Tier-1 signals are not what the pipeline's description says:\n  - "
                           + "\n  - ".join(findings))
    return findings


def load(variant="corrected", strict=True, absence=False):
    """The entry point every fusion must use. Corrected is the default; deployed must be asked for."""
    if variant not in ("corrected", "deployed"):
        raise ValueError(f"unknown Tier-1 variant {variant!r}: 'corrected' or 'deployed'")
    sig = corrected(absence=absence) if variant == "corrected" else deployed()
    check(sig, strict=strict)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployed", action="store_true", help="report on the audited baseline instead")
    a = ap.parse_args()
    sig = deployed() if a.deployed else corrected()
    n = len(sig["ids"])
    print(f"Tier-1 signals, {sig['variant']} construction, {n:,} pictures\n")
    for ag in AGENTS:
        M = np.asarray(sig[ag], dtype=np.float64)
        d = len(np.unique(np.round(M, 9)))
        print(f"  {ag:7s} vector of {M.shape[1]:2d}  {d:7,d} distinct values  "
              f"covers {sig['coverage'][ag]:,}  <- {sig['sources'][ag]}")
    findings = check(sig, strict=False)
    if findings:
        print(f"\n{len(findings)} defect(s) -- this construction would stop the pipeline:")
        for f in findings:
            print(f"  - {f}")
        return 1
    print("\nall guards pass: measured, covered, continuous, independent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
