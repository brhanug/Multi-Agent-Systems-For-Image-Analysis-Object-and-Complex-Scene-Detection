#!/usr/bin/env python3
"""Does the risk guarantee hide its own failure? A class-conditional reformulation.

The deployed guarantee controls the MARGINAL risk:

    R(lambda) = (1/n) sum_j  error_j * 1[U_j < lambda]     <= alpha

averaged over every image. The evaluation pool is 84.5% scene-positive and the system is
documented as near-blind on true negatives, so that average is dominated by the class the
system handles well. A marginal bound can therefore be satisfied while the risk on the
negative class -- the one a curator most needs flagged -- is several times higher.

This tests the obvious mathematical repair: Mondrian, or class-conditional, conformal risk
control. Instead of one threshold, calibrate a separate lambda within each group and
require the bound to hold in each:

    R_g(lambda_g) = (1/n_g) sum_{j in g}  error_j * 1[U_j < lambda_g]   <= alpha  for all g

The grouping must use only what is known at deployment time, so images are grouped by what
the system PREDICTS, not by the truth. Two groupings are tested: the predicted class, and
the negative-evidence score, which is available for every image before any label exists.

Three questions:
  1. Under the deployed marginal bound, what is the risk inside each group?
  2. Does conditioning restore the guarantee where it is broken?
  3. What does it cost in automatic approval?

Outputs: results/signal_rebuild/conditional_risk_control.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
NEG = SR / "negative_evidence_scores.npz"
OUT = SR / "conditional_risk_control.json"
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]
GRID = np.arange(0.02, 0.98, 0.002)
LGRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]


def rj(paths):
    r = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(r).drop_duplicates("image_id")


def bal_acc(pred, h):
    tp = int(((pred == 1) & (h == 1)).sum()); fn = int(((pred == 0) & (h == 1)).sum())
    tn = int(((pred == 0) & (h == 0)).sum()); fp = int(((pred == 1) & (h == 0)).sum())
    return ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2


def crc_lambda(U_cal, E_cal, alpha):
    """Largest lambda whose finite-sample corrected calibration risk stays within alpha."""
    n = len(U_cal)
    if n == 0:
        return float(LGRID[0])
    risks = np.array([np.mean(E_cal * (U_cal < g)) for g in LGRID])
    ok = (n / (n + 1.0)) * risks + (1.0 / (n + 1.0)) <= alpha
    return float(np.max(LGRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing


def measure(U, E, lam):
    auto = U < lam
    n = len(U)
    return {"n": int(n),
            "auto_approved": int(auto.sum()),
            "auto_approval_pct": round(100 * float(auto.mean()), 2) if n else None,
            "risk": round(float(np.mean(E * auto)), 4) if n else None,
            "errors_let_through": int((E * auto).sum()),
            "errors_total": int(E.sum())}


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    s["A_scene"] = s.max_sigmoid.astype(float)          # the corrected channel
    y["A_object"] = y.conf_max.clip(0, 1)
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm"] = txt.apply(lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAX) / len(TAX))
    d = (y[["image_id", "A_object"]].merge(s[["image_id", "A_scene"]], on="image_id")
         .merge(v[["image_id", "A_vlm"]], on="image_id"))
    z = np.load(NEG, allow_pickle=True)
    d = d.merge(pd.DataFrame({"image_id": list(z["ids"]), "neg_evidence": z["neg"]}),
                on="image_id", how="left")
    d["neg_evidence"] = d.neg_evidence.fillna(d.neg_evidence.median())
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--split-csv", default=None); ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    global SPLIT, OUT
    if args.split_csv: SPLIT = Path(args.split_csv)
    if args.out_json: OUT = Path(args.out_json)
    d = build()
    j = pd.read_csv(SPLIT).merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    tr, cal, te = j[j.split == "train"], j[j.split == "calibration"], j[j.split == "test"]

    # the best available classifier: corrected channel, weights fitted on train
    cols = ["A_object", "A_scene", "A_vlm"]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[cols].values, tr.has_scene.values)
    w = lr.coef_[0] / np.abs(lr.coef_[0]).sum()
    saa = lambda f: f[cols].values @ w
    thr = float(GRID[int(np.argmax([bal_acc((saa(cal) >= t).astype(int), cal.has_scene.values)
                                    for t in GRID]))])
    def parts(f):
        sc = saa(f); pred = (sc >= thr).astype(int)
        return sc, pred, (pred != f.has_scene.values).astype(int)
    sc_c, pr_c, e_c = parts(cal)
    sc_t, pr_t, e_t = parts(te)
    U_c, U_t = 1 - sc_c, 1 - sc_t          # triage on confidence, per Section 2.4

    out = {"experiment": "Class-conditional (Mondrian) conformal risk control",
           "weights": {"object": round(float(w[0]), 4), "scene": round(float(w[1]), 4),
                       "vlm": round(float(w[2]), 4)},
           "decision_threshold": round(thr, 4),
           "test_n": int(len(te)), "test_errors": int(e_t.sum()),
           "test_predicted_positive": int(pr_t.sum()), "test_predicted_negative": int((1-pr_t).sum())}

    # ---- 1. what the marginal bound actually delivers, group by group --------------
    marginal = []
    for a in ALPHAS:
        lam = crc_lambda(U_c, e_c, a)
        row = {"alpha": a, "lambda": round(lam, 4), "overall": measure(U_t, e_t, lam)}
        row["overall"]["bound_satisfied"] = bool(row["overall"]["risk"] <= a)
        for gname, m in [("predicted_positive", pr_t == 1), ("predicted_negative", pr_t == 0)]:
            g = measure(U_t[m], e_t[m], lam)
            g["bound_satisfied_within_group"] = bool(g["risk"] is not None and g["risk"] <= a)
            row[gname] = g
        marginal.append(row)
    out["marginal_bound_as_deployed"] = marginal
    # A marginal bound can be met while EITHER group exceeds its own target; check both
    # rather than only the group the blind-spot finding made me expect.
    hidden = []
    for r in marginal:
        if not r["overall"]["bound_satisfied"]:
            continue
        for gname in ("predicted_positive", "predicted_negative"):
            g = r[gname]
            if g["risk"] is not None and not g["bound_satisfied_within_group"]:
                hidden.append({"alpha": r["alpha"], "group": gname,
                               "group_risk": g["risk"], "target": r["alpha"],
                               "exceeds_target_by_pct": round(100 * (g["risk"] - r["alpha"]) / r["alpha"], 1),
                               "of_that_group_auto_approved_pct": g["auto_approval_pct"]})
    out["marginal_bound_met_while_a_group_exceeds"] = hidden

    # ---- 2. the class-conditional reformulation ------------------------------------
    cond = []
    for a in ALPHAS:
        row = {"alpha": a, "per_group": {}}
        auto_n = 0
        for gname, mc, mt in [("predicted_positive", pr_c == 1, pr_t == 1),
                              ("predicted_negative", pr_c == 0, pr_t == 0)]:
            lam_g = crc_lambda(U_c[mc], e_c[mc], a)
            g = measure(U_t[mt], e_t[mt], lam_g)
            g["lambda"] = round(lam_g, 4)
            g["bound_satisfied_within_group"] = bool(g["risk"] is not None and g["risk"] <= a)
            row["per_group"][gname] = g
            auto_n += g["auto_approved"]
        row["overall_auto_approval_pct"] = round(100 * auto_n / len(te), 2)
        row["all_groups_satisfied"] = all(v["bound_satisfied_within_group"] for v in row["per_group"].values())
        cond.append(row)
    out["class_conditional_bound"] = cond

    # ---- 3. grouping by negative evidence, which needs no prediction ----------------
    cut = float(np.median(cal.neg_evidence.values))
    negcond = []
    for a in ALPHAS:
        row = {"alpha": a, "split_at_negative_evidence": round(cut, 4), "per_group": {}}
        auto_n = 0
        for gname, mc, mt in [("low_negative_evidence", cal.neg_evidence.values < cut,
                               te.neg_evidence.values < cut),
                              ("high_negative_evidence", cal.neg_evidence.values >= cut,
                               te.neg_evidence.values >= cut)]:
            lam_g = crc_lambda(U_c[mc], e_c[mc], a)
            g = measure(U_t[mt], e_t[mt], lam_g)
            g["lambda"] = round(lam_g, 4)
            g["bound_satisfied_within_group"] = bool(g["risk"] is not None and g["risk"] <= a)
            row["per_group"][gname] = g
            auto_n += g["auto_approved"]
        row["overall_auto_approval_pct"] = round(100 * auto_n / len(te), 2)
        row["all_groups_satisfied"] = all(v["bound_satisfied_within_group"] for v in row["per_group"].values())
        negcond.append(row)
    out["conditioned_on_negative_evidence"] = negcond

    # ---- 4. what it costs -----------------------------------------------------------
    out["cost_of_conditioning"] = [
        {"alpha": m["alpha"],
         "marginal_auto_approval_pct": m["overall"]["auto_approval_pct"],
         "class_conditional_auto_approval_pct": c["overall_auto_approval_pct"],
         "difference_pp": round(c["overall_auto_approval_pct"] - m["overall"]["auto_approval_pct"], 2)}
        for m, c in zip(marginal, cond)]

    out["reading"] = (
        "The marginal bound averages over a pool that is 84.5% scene-positive, so it is "
        "dominated by the class the system handles well. The rows to read are the "
        "predicted_negative groups: where the overall bound is satisfied but that group's is "
        "not, the deployed guarantee is being met on paper while the class a curator most "
        "needs protected is above its target. Conditioning is the standard repair and costs "
        "automatic approval; both numbers are given so the trade is explicit.")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
