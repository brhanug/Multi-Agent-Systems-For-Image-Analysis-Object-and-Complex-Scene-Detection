#!/usr/bin/env python3
"""The running Agent 2 scored on the 300 hand-annotated pictures (self_annotation_package/more_experiments/
PLAN_agent2.md, commit 6c018af).

Agent 2 as run on the archive (run_tier2_tier3_from_full_tier1.py): the VLM's free-text answers mapped to
the detector taxonomy by a word list, a flag for each class named by the VLM and not returned by the
detector at confidence 0.25. Scored on the eight shared classes against the human annotation, with the
deployed detector and with the retrained object finder.

    python run_agent2_on_300.py [--verify]
"""
import argparse, glob, json, re

import numpy as np
import pandas as pd

import run_best_config_repeated as rb
import run_object_classes as ro
from run_tier2_tier3_from_full_tier1 import TAXONOMY, read_jsonl, SR

OUT = rb.SR / "agent2_on_300.json"
EIGHT = ro.CLASSES


def claimed(text):
    out = set()
    for cls, forms in TAXONOMY.items():
        if any(re.search(rf"\b{re.escape(f)}\b", text) for f in forms):
            out.add("animal" if cls == "horse" else cls)
    return out & set(EIGHT)


def wilson(k, n, z=1.959964):
    if not n:
        return [None, None]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def score(claims, detected, truth):
    flags = hall = miss = 0; per = {c: [0, 0] for c in EIGHT}; vlm_claims = vlm_hall = vlm_hall_flagged = 0
    for pid in truth.index:
        cl, det = claims.get(pid, set()), detected.get(pid, set())
        for c in cl:
            absent = int(truth.at[pid, c] == 0)
            vlm_claims += 1; vlm_hall += absent
            if c not in det:
                flags += 1; hall += absent; miss += (not absent); per[c][0] += 1; per[c][1] += absent
                vlm_hall_flagged += absent
    return {"flags": flags, "pictures_with_a_flag": int(sum(bool(claims.get(p, set()) - detected.get(p, set())) for p in truth.index)),
            "hallucinations_among_flags": hall, "detector_misses_among_flags": miss,
            "precision": round(hall / flags, 4) if flags else None, "precision_wilson95": wilson(hall, flags),
            "vlm_claims": vlm_claims, "vlm_hallucinations": vlm_hall,
            "base_rate_hallucination_among_claims": round(vlm_hall / vlm_claims, 4) if vlm_claims else None,
            "recall_of_vlm_hallucinations": round(vlm_hall_flagged / vlm_hall, 4) if vlm_hall else None,
            "per_class": {c: {"flags": f, "hallucinations": h} for c, (f, h) in per.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    truth = pd.read_csv(ro.CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")[EIGHT].astype(int)
    llava = read_jsonl(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    llava = llava[llava.image_id.isin(truth.index)]
    claims = {r.image_id: claimed(" ".join(str(x) for x in r.answers).lower() if isinstance(r.answers, list) else "")
              for r in llava.itertuples()}
    yolo = read_jsonl([SR / "yolo_full_corpus.jsonl"]); yolo = yolo[yolo.image_id.isin(truth.index)]
    deployed = {r.image_id: {"animal" if x == "horse" else x for x in (r.labels if isinstance(r.labels, list) else [])}
                for r in yolo.itertuples()}
    retrained = {}
    for line in open(ro.CS / "new_student_class_scores.jsonl"):
        r = json.loads(line)
        if r["image_id"] in truth.index:
            retrained[r["image_id"]] = {c for c, n in r["count_at_0.25"].items() if n > 0 and c in EIGHT}
    missing = [p for p in truth.index if p not in claims or p not in deployed or p not in retrained]
    assert not missing, f"{len(missing)} pictures lack outputs"
    # extra, after the planned run: claims of absent classes whose every mention is negated or hedged
    NEG = re.compile(r"\b(no|not|without|nor|absence of|lack of|neither|n't|any)\b[^.]{0,60}$")
    HEDGE = re.compile(r"\b(possibly|perhaps|might|may be|could be|appears? to be|seems?|likely|what looks like|suggest)\b[^.]{0,40}$")
    neg = hed = tot = 0
    for r in llava.itertuples():
        t = " ".join(str(x) for x in r.answers).lower() if isinstance(r.answers, list) else ""
        for cls, forms in TAXONOMY.items():
            c = "animal" if cls == "horse" else cls
            if c not in EIGHT or truth.at[r.image_id, c] != 0:
                continue
            ms = [m for f in forms for m in re.finditer(rf"\b{re.escape(f)}\b", t)]
            if not ms:
                continue
            tot += 1
            if all(NEG.search(t[max(0, m.start() - 80):m.start()]) for m in ms): neg += 1
            elif all(HEDGE.search(t[max(0, m.start() - 60):m.start()]) for m in ms): hed += 1
    out = {"plan": "self_annotation_package/more_experiments/PLAN_agent2.md (commit 6c018af)", "pictures": int(len(truth)),
           "extra_claims_of_absent_classes": {"claims": tot, "all_mentions_negated": neg, "all_mentions_hedged": hed},
           "deployed detector": score(claims, deployed, truth), "retrained object finder": score(claims, retrained, truth)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
