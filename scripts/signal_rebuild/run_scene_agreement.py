#!/usr/bin/env python3
"""Scene-level agreement between the systems, beside the agreement between two people.

The object-level version (run_class_agreement.py) covers the eight frozen object classes. This does
the same for the five scene types, with the two systems that score every scene type on every picture:
SigLIP (sigmoid over the five scene prompts) and LLaVA-OneVision (one yes/no question per scene type,
read as P(yes); run_llava_scene_scores.py). The text reader does not score scenes and the detector
has no scene classes, so there is one machine pair per scene type.

  continuous  agreement = 1 - |percentile_SigLIP - percentile_LLaVA| per picture and scene type,
              on within-corpus percentiles (the raw scales differ), over all pictures
  decisions   each system's cut-off per scene type is the Youden point on the 300 hand-labelled
              pictures; over the corpus, Cohen's kappa, raw agreement, PABAK and prevalence
  vs people   on the 300, each system against the rater (cut-offs cross-fitted over five folds),
              the machine pair on the same pictures, and AUROC of each system per scene type
  machines vs people  on the 124 pictures two people rated, the machine pair's kappa beside the
              two people's kappa, per scene type

    python run_scene_agreement.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "class_agreement/scene_agreement.json"
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
SYSTEMS = ["siglip", "llava"]
RATER = BASE / "human_spatial_audit/user_annotations_300.csv"
HUMAN_KAPPA = BASE / "results/multi_agent/per_class_human_kappa.json"
SEED = 20261006


def norm(i):
    i = str(i)
    return i if "/" in i else i.replace("_", "/", 1)


def youden(v, y):
    fpr, tpr, th = roc_curve(y, v)
    return float(th[np.argmax(tpr - fpr)])


def kappa(a, b):
    return None if len(set(a)) < 2 and len(set(b)) < 2 else round(float(cohen_kappa_score(a, b)), 4)


def pct(x):
    return pd.Series(x).rank(pct=True).values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    sig = {}
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sig[norm(r["image_id"])] = [float(r["sigmoid"].get(s, 0.0)) for s in SCENES]
    lla = {}
    for f in sorted(glob.glob(str(SR / "class_scores/llava_scene_scores/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); lla[norm(r["image_id"])] = [float(r["p_yes"][s]) for s in SCENES]
    ids = sorted(set(sig) & set(lla)); pos = {i: k for k, i in enumerate(ids)}
    R = np.stack([np.array([sig[i] for i in ids]), np.array([lla[i] for i in ids])], axis=1)   # pictures x systems x scenes
    P = np.stack([np.stack([pct(R[:, s, j]) for j in range(len(SCENES))], axis=1) for s in range(2)], axis=1)

    rater = pd.read_csv(RATER, keep_default_na=False)
    rater["iid"] = rater.image_id.map(norm)
    rater = rater[rater.iid.isin(pos)].drop_duplicates("iid")
    rows = rater.iid.map(pos).values
    truth = {s: (rater.scene.str.strip().str.lower() == s).astype(int).values for s in SCENES}

    A = 1 - np.abs(P[:, 0] - P[:, 1])                                             # pictures x scenes
    cont = {"per_scene": {s: round(float(A[:, j].mean()), 4) for j, s in enumerate(SCENES)},
            "global_mean": round(float(A.mean()), 4),
            "chance_level_note": "two independent uniform percentiles agree 0.667 on average"}

    D = np.zeros(R.shape, dtype=int)
    cut = {}
    for s in range(2):
        for j, c in enumerate(SCENES):
            t = youden(R[rows, s, j], truth[c]); cut[(s, c)] = t
            D[:, s, j] = (R[:, s, j] >= t).astype(int)
    corpus = {}
    for j, c in enumerate(SCENES):
        po = float((D[:, 0, j] == D[:, 1, j]).mean())
        corpus[c] = {"prevalence": {SYSTEMS[s]: round(float(D[:, s, j].mean()), 4) for s in range(2)},
                     "cohen_kappa": kappa(D[:, 0, j], D[:, 1, j]), "raw_agreement": round(po, 4), "pabak": round(2 * po - 1, 4)}

    Dh = np.zeros((len(rows), 2, len(SCENES)), dtype=int)
    for j, c in enumerate(SCENES):
        y = truth[c]
        for s in range(2):
            v = R[rows, s, j]
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(v, y):
                Dh[te, s, j] = (v[te] >= youden(v[tr], y[tr])).astype(int)
    on300 = {}
    for j, c in enumerate(SCENES):
        y = truth[c]
        on300[c] = {"rater_positives": int(y.sum()),
                    "auroc": {SYSTEMS[s]: round(float(roc_auc_score(y, R[rows, s, j])), 4) for s in range(2)},
                    "system_vs_rater_kappa": {SYSTEMS[s]: kappa(Dh[:, s, j], y) for s in range(2)},
                    "machine_pair_kappa": kappa(Dh[:, 0, j], Dh[:, 1, j])}

    hk = json.loads(HUMAN_KAPPA.read_text())
    per = hk.get("per_category") or hk.get("categories") or []
    human = {r["category"]: r for r in per} if isinstance(per, list) else per
    c1 = pd.read_csv(BASE / "human_spatial_audit/user_annotations_800.csv")
    c1["cvat_id"] = c1.index
    c1["full_id"] = c1["raw_name"].str.replace("images/", "", regex=False).str.replace(".jpg", "", regex=False)
    both = set(c1[c1.cvat_id <= 800].full_id.map(norm))
    sel = np.array([i in both for i in rater.iid])
    vs_people = {}
    for j, c in enumerate(SCENES):
        h = human.get(c, {})
        vs_people[c] = {"pictures": int(sel.sum()),
                        "two_people_kappa": h.get("cohens_kappa"), "two_people_ci95": h.get("ci95"),
                        "machine_pair_kappa": kappa(Dh[sel, 0, j], Dh[sel, 1, j]),
                        "siglip_vs_rater_kappa": kappa(Dh[sel, 0, j], truth[c][sel]),
                        "llava_vs_rater_kappa": kappa(Dh[sel, 1, j], truth[c][sel])}

    out = {"experiment": "Scene-level agreement between SigLIP and LLaVA, beside agreement between two people",
           "pictures": len(ids), "scenes": SCENES, "systems": SYSTEMS, "continuous_agreement": cont,
           "decisions_corpus": corpus, "on_300": on300, "machines_vs_people_on_double_rated": vs_people}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    print(f"{len(ids)} pictures with both systems; 300-rater rows matched: {len(rows)}; double-rated: {int(sel.sum())}\n")
    print("continuous agreement per scene:", cont["per_scene"], "| global", cont["global_mean"])
    print(f"\n{'scene':10s} {'pos':>4s} {'AUROC sig/lla':>14s} {'kappa sig/lla vs rater':>23s} {'pair(300)':>9s} {'pair(corpus)':>12s} {'PABAK':>6s}")
    for c in SCENES:
        o, k = on300[c], corpus[c]
        print(f"{c:10s} {o['rater_positives']:4d} {o['auroc']['siglip']:6.3f}/{o['auroc']['llava']:.3f} "
              f"{str(o['system_vs_rater_kappa']['siglip']):>10s}/{str(o['system_vs_rater_kappa']['llava']):<10s} "
              f"{str(o['machine_pair_kappa']):>9s} {str(k['cohen_kappa']):>12s} {k['pabak']:6.3f}")
    print("\nmachines vs people on the double-rated pictures (kappa)")
    for c, v in vs_people.items():
        print(f"  {c:10s} two people {v['two_people_kappa']}  machine pair {v['machine_pair_kappa']}  "
              f"siglip-rater {v['siglip_vs_rater_kappa']}  llava-rater {v['llava_vs_rater_kappa']}")


if __name__ == "__main__":
    raise SystemExit(main())
