#!/usr/bin/env python3
"""AR3, AR4, AR8: the read-out of each model is what the corrected pipeline assumes it is.

The Tier-1 guards check the signals' shape -- measured, covered, continuous, independent. These
checks look one step earlier, at how each model's raw output was turned into a number, where the
September audit found every defect:

  AR3 scene    every SigLIP scorer pads text to max_length (other padding silently degrades SigLIP)
               and applies the model's own logit scale and bias before a sigmoid
  AR4 VLM      P(yes) counts every top-20 candidate that decodes to yes or no after stripping and
               lower-casing, so leading-space and capitalised variants all count; decoding is greedy
               with one token; and no answer is left undefined (neither word in the top 20)
  AR8 ties     no decision threshold sits on a reachable score value: the Level-2 threshold of 0.5
               is hit by no picture, and the corrected confidence score is continuous enough that
               the risk-control grid never has to split a tie

Each check fails loudly. Run after any change to a scorer or to the Tier-1 construction.

    python check_agent_readouts.py
"""
import glob, json, re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SRC = BASE / "scripts/signal_rebuild"
SR = BASE / "results/signal_rebuild"


def ar3():
    out = []
    for name in ("run_siglip_class_scores.py", "run_siglip_scene_full_corpus.py", "run_negative_evidence_detector.py"):
        t = (SRC / name).read_text()
        pads = re.findall(r"padding=\"([a-z_]+)\"", t)
        assert pads and set(pads) == {"max_length"}, f"{name}: text padding {pads}, must be max_length"
        assert "logit_scale" in t and "logit_bias" in t, f"{name}: does not apply SigLIP's own scale and bias"
        out.append(f"{name}: padding max_length, own scale and bias")
    t = (SRC / "tier1_signals.py").read_text()
    assert '"sigmoid"' in t and '"softmax"' not in t, "tier1_signals reads the softmax channel"
    out.append("tier1_signals reads the sigmoid channel only")
    return out


def ar4():
    t = (SRC / "run_llava_class_scores.py").read_text()
    assert "temperature=0.0" in t and "max_tokens=1" in t, "LLaVA decoding is not greedy with one token"
    assert ".strip().lower()" in t and 'tok == "yes"' in t and 'tok == "no"' in t, \
        "P(yes) does not normalise yes/no variants"
    total = undefined = 0
    for f in sorted(glob.glob(str(SR / "class_scores/llava_class_scores/shard_*.jsonl"))):
        for line in open(f):
            for v in json.loads(line)["p_yes"].values():
                total += 1; undefined += v is None
    assert total == 12110 * 12, f"LLaVA answers {total}, expected {12110 * 12}"
    assert undefined == 0, f"{undefined} LLaVA answers had neither yes nor no in the top 20"
    return [f"greedy, one token, yes/no variants normalised; {total:,} answers, none undefined"]


def ar8():
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(BASE / "results/multi_agent/corrected_clean_holdout_split.csv")
    tr = sp[(sp.split == "train") & sp.ppn_id.isin(idx)]
    V = {**nat, "scene": scene_abs}
    allr = np.arange(len(idx))
    P = np.column_stack([e2e.to_probability(V[k], tr.ppn_id.map(idx).values, tr.has_scene.values, allr)
                         for k in ("object", "scene", "vlm")])
    saa = P @ np.array([0.6, 0.2, 0.2])
    conf = 1 - np.abs(saa - 0.5)
    on_threshold = int(np.sum(saa == 0.5))
    assert on_threshold == 0, f"{on_threshold} pictures sit exactly on the 0.5 decision threshold"
    grid = np.linspace(0.001, 1.0, 1000)
    on_grid = int(np.isin(np.round(conf, 12), np.round(grid, 12)).sum())
    distinct = len(np.unique(conf))
    assert distinct > 0.99 * len(conf), f"confidence takes only {distinct:,} values over {len(conf):,} pictures"
    assert on_grid == 0, f"{on_grid} confidence values equal a risk-control grid point"
    return [f"no picture on the 0.5 threshold; confidence has {distinct:,} distinct values over "
            f"{len(conf):,} pictures, none on the risk-control grid"]


def main():
    import sys
    sys.path.insert(0, str(SRC))
    failed = 0
    for tag, fn in (("AR3 scene", ar3), ("AR4 VLM", ar4), ("AR8 ties", ar8)):
        try:
            for line in fn(): print(f"  ok      {tag:10s} {line}")
        except AssertionError as e:
            failed += 1; print(f"  FAILED  {tag:10s} {e}")
    print("\nall read-out checks pass" if not failed else f"\n{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
