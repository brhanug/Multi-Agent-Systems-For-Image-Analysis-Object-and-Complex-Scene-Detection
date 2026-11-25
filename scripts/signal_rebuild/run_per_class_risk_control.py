#!/usr/bin/env python3
"""E3, second part: risk control per object class instead of per picture.

The deployed guarantee approves a whole picture. An archive would rather approve the class labels it
can trust and review the rest: keep "person" automatic and send "weapon" to a human. That needs one
threshold per class, calibrated per class.

Data: the 300 hand-annotated pictures, the only ones with object labels. Each (picture, class) pair is
one decision. The score is the fused per-class probability of the three systems, Platt-calibrated;
the uncertainty of a decision is its distance from the decision boundary. Calibration and test split
by book (50/50), repeated over 200 draws, with calibration fitted inside the calibration books only.

Compared per draw:
  one threshold for all classes   -- the deployed shape of the guarantee
  one threshold per class         -- the per-class version

Reported per class: how much is auto-approved, the risk among all its pairs, the error rate among
the approved ones, and of the class's objects the share found without a person (approved as present)
and the share missed without anyone looking (approved as absent). With 300 pictures and few positives for rare classes this is a demonstration of the
mechanism, not a deployment figure.

    python run_per_class_risk_control.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/c4/per_class_risk_control.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SYSTEMS = ["detector", "siglip", "llava"]
ALPHAS, N_DRAWS, SEED = [0.01, 0.02, 0.05, 0.10], 200, 20260920
GRID = np.linspace(0.001, 1.0, 1000)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def fused_probability(R, truth, cal_rows, all_rows):
    """Platt scaling per system and class, fitted on the calibration books only."""
    P = np.zeros((len(all_rows), len(SYSTEMS), len(CLASSES)))
    for si in range(len(SYSTEMS)):
        for ci, c in enumerate(CLASSES):
            y = truth[c].values[cal_rows]; x = logit(R[cal_rows, si, ci]).reshape(-1, 1)
            if 0 < y.sum() < len(y):
                m = LogisticRegression(C=100.0).fit(x, y)
                P[:, si, ci] = m.predict_proba(logit(R[all_rows, si, ci]).reshape(-1, 1))[:, 1]
            else:
                P[:, si, ci] = float(y.mean())
    return P.mean(axis=1)


def lam_for(u, e, alpha, groups=None):
    """Largest threshold meeting the finite-sample bound. With groups, the unit is the group (book),
    not the single decision: decisions inside one picture and one book are correlated, and treating
    them as independent makes the bound too loose."""
    n = len(u)
    if n == 0: return GRID[0]
    loss = e[None, :] * (u[None, :] < GRID[:, None])
    if groups is None:
        risk = loss.mean(axis=1); n_units = n
    else:
        gs = np.unique(groups)
        risk = np.stack([loss[:, groups == g].mean(axis=1) for g in gs], axis=1).mean(axis=1); n_units = len(gs)
    ok = (n_units / (n_units + 1)) * risk + 1 / (n_units + 1) <= alpha
    return GRID[ok].max() if ok.any() else 0.0


def summarise(acc):
    # the standard error of the mean risk over draws is added (23 Nov 2026); draws share pictures, so it is
    # a lower bound on the uncertainty of that mean, reported only to size the small excesses over alpha
    return {p: {str(al): {c: {**{q: round(float(np.mean(v[q])), 4) for q in v},
                              "risk_se_over_draws": round(float(np.std(v["risk"], ddof=1) / np.sqrt(len(v["risk"]))), 4),
                              "share_of_draws_above_alpha": round(float(np.mean(np.array(v["risk"]) > al)), 3)}
                          for c, v in byc.items()}
                 for al, byc in byal.items()} for p, byal in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    R = m["raw"][:, [ms.index(s) for s in SYSTEMS]][:, :, [mc.index(c) for c in CLASSES]].astype(np.float64)
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    # the corrected design's object finder is the rebuilt student (Section 6.13): swap its per-class scores in
    Rn = R.copy()
    for line in open(CS / "new_student_class_scores.jsonl"):
        r = json.loads(line); k = pos.get(r["image_id"])
        if k is None: continue
        for c, v in r["max_conf"].items():
            if c in CLASSES: Rn[k, SYSTEMS.index("detector"), CLASSES.index(c)] = v
    rows = np.array([pos[i] for i in truth.index]); tt = truth.reset_index(drop=True)
    books = np.array([i.split("/")[0] for i in truth.index]); uniq = np.unique(books)
    def run(M):
        rng = np.random.default_rng(SEED)
        acc = {p: {al: {c: {"approved": [], "risk": [], "risk_among_approved": [], "objects_found": [],
                                 "objects_missed_and_approved": []} for c in CLASSES} for al in ALPHAS}
               for p in ("one_threshold", "per_class", "per_class_by_book")}
        for _ in range(N_DRAWS):
            cal_books = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
            cal = np.array([b in cal_books for b in books]); test = ~cal
            P = fused_probability(M, tt, np.where(cal)[0], np.arange(len(rows)))
            pred = (P >= 0.5).astype(int)
            err = (pred != tt[CLASSES].values).astype(int)
            unc = 1 - np.abs(P - 0.5)
            for al in ALPHAS:
                lam_all = lam_for(unc[cal].ravel(), err[cal].ravel(), al)
                lam_c = {ci: lam_for(unc[cal, ci], err[cal, ci], al) for ci in range(len(CLASSES))}
                lam_b = {ci: lam_for(unc[cal, ci], err[cal, ci], al, groups=books[cal]) for ci in range(len(CLASSES))}
                for policy, thr in (("one_threshold", {ci: lam_all for ci in range(len(CLASSES))}), ("per_class", lam_c),
                                    ("per_class_by_book", lam_b)):
                    for ci, c in enumerate(CLASSES):
                        auto = unc[test, ci] < thr[ci]; e = err[test, ci]
                        k = acc[policy][al][c]
                        k["approved"].append(float(auto.mean()))
                        k["risk"].append(float((e * auto).mean()))
                        k["risk_among_approved"].append(float(e[auto].mean()) if auto.any() else 0.0)
                        y = tt[c].values[test]; npos = max(1, int(y.sum()))
                        # of the class's objects in the test pictures: how many are approved as present (found
                        # without a person), and how many are approved as absent (missed, and nobody looks)
                        k["objects_found"].append(float(((pred[test, ci] == 1) & auto & (y == 1)).sum() / npos))
                        k["objects_missed_and_approved"].append(float(((pred[test, ci] == 0) & auto & (y == 1)).sum() / npos))
        return acc
    acc = run(R); acc_n = run(Rn)
    out = {"experiment": "Conformal risk control per object class (E3, second part)",
           "pictures": int(len(rows)), "decisions_per_draw": int(len(rows) * len(CLASSES) // 2),
           "draws": N_DRAWS, "split": "by book, half calibration half test",
           "score": "fused per-class probability of detector, SigLIP and LLaVA, Platt-calibrated on the calibration books",
           "positives_in_300": {c: int(truth[c].sum()) for c in CLASSES},
           "policies": summarise(acc),
           "policies_rebuilt_detector": summarise(acc_n),
           "note": "policies: the deployed detector beside SigLIP and LLaVA; policies_rebuilt_detector: the corrected design's object finder (the rebuilt student) in its place"}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    for al in ALPHAS:
        print(f"\ntarget {al}: class risk and approved share -- one threshold / per class / per class by book")
        print("  class")
        for c in CLASSES:
            o = out["policies"]["one_threshold"][str(al)][c]; p = out["policies"]["per_class"][str(al)][c]
            b = out["policies"]["per_class_by_book"][str(al)][c]
            print(f"  {c:9s} risk {o['risk']:.4f} / {p['risk']:.4f} / {b['risk']:.4f}   approved {100*o['approved']:5.1f}% / {100*p['approved']:5.1f}% / {100*b['approved']:5.1f}%")
        oa = np.mean([out["policies"]["one_threshold"][str(al)][c]["approved"] for c in CLASSES])
        pa = np.mean([out["policies"]["per_class"][str(al)][c]["approved"] for c in CLASSES])
        print(f"  mean approved: one threshold {100*oa:.1f}%   per class {100*pa:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
