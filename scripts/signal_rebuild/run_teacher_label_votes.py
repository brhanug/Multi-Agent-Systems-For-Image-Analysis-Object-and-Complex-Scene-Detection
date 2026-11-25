#!/usr/bin/env python3
"""How many of the other systems agree with each pseudo-label (Phase 0, B2 -- part 2).

A pseudo-label says "class c is in this picture". Each other system votes yes when its
per-class score clears a cut-off. SigLIP and LLaVA give picture-level scores, so the
vote is picture-level; the detector votes with its own top box confidence for c.
Kosmos-2.5 is not a voter: neither pseudo-label set contains a text label.

Cut-offs are set per class and system where that system best separates the 300
human-annotated pictures (Youden's J). Classes with no human positives (horse,
clothing) get no cut-off and are reported without votes.

The detector was trained on v1_refresh, so its vote on that set is not independent;
agreement is reported with and without it.

    python run_teacher_label_votes.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = CS / "teacher_label_votes.json"
SETS = {"final_dataset": ([BASE / "final_dataset/labels/yolo_labels"],
                          ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]),
        "v1_refresh": ([BASE / f"data/yolo11_v1_refresh/labels/{s}" for s in ("train", "val", "test")],
                       ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = list(m["ids"]); pos = {i: k for k, i in enumerate(ids)}
    classes, systems, R = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]], m["raw"].astype(np.float64)
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows300 = [pos[i] for i in truth.index]
    cut = {}
    for si, s in enumerate(systems):
        for ci, c in enumerate(classes):
            y, v = truth[c].values, R[rows300, si, ci]
            if 0 < y.sum() < len(y) and not np.isnan(v).any():
                fpr, tpr, th = roc_curve(y, v); cut[(s, c)] = float(th[np.argmax(tpr - fpr)])
    out = {"experiment": "Votes of the other systems on each pseudo-label (B2, part 2)",
           "cutoffs": {f"{s}/{c}": round(v, 5) for (s, c), v in sorted(cut.items())}, "sets": {}}
    for name, (dirs, names) in SETS.items():
        pairs = []
        for f in (f for d in dirs for f in glob.glob(str(d / "*.txt"))):
            k = pos.get(Path(f).stem.replace("_", "/", 1))
            if k is None: continue
            for cid in {int(float(l.split()[0])) for l in open(f) if l.strip()}:
                pairs.append((k, names[cid]))
        per_class = []
        for c in sorted({c for _, c in pairs}, key=lambda c: names.index(c)):
            ks = np.array([k for k, cc in pairs if cc == c]); ci = classes.index(c)
            voters = [s for s in systems if (s, c) in cut]
            entry = {"class": c, "pseudo_labels": int(len(ks)), "voters": voters}
            if voters:
                votes = {s: R[ks, systems.index(s), ci] >= cut[(s, c)] for s in voters}
                n = sum(v.astype(int) for v in votes.values())
                entry["share_each_system_agrees"] = {s: round(float(v.mean()), 3) for s, v in votes.items()}
                entry["histogram_number_agreeing"] = {str(j): int((n == j).sum()) for j in range(len(voters) + 1)}
                indep = [s for s in voters if not (name == "v1_refresh" and s == "detector")]
                ni = sum(votes[s].astype(int) for s in indep)
                entry["share_no_independent_system_agrees"] = round(float((ni == 0).mean()), 3)
                entry["share_all_independent_systems_agree"] = round(float((ni == len(indep)).mean()), 3)
            else:
                entry["note"] = "no human positives in the 300, so no cut-off; not voted"
            per_class.append(entry)
        out["sets"][name] = {"pseudo_label_pairs": len(pairs), "per_class": per_class}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for name, s in out["sets"].items():
        print(f"\n{name}: {s['pseudo_label_pairs']} picture-class pseudo-labels")
        for e in s["per_class"]:
            if "note" in e: print(f"  {e['class']:9s} {e['pseudo_labels']:5d}  {e['note']}"); continue
            print(f"  {e['class']:9s} {e['pseudo_labels']:5d}  agree: {e['share_each_system_agrees']}  none(indep) {e['share_no_independent_system_agrees']:.2f}  all(indep) {e['share_all_independent_systems_agree']:.2f}  hist {e['histogram_number_agreeing']}")


if __name__ == "__main__":
    raise SystemExit(main())
