#!/usr/bin/env python3
"""Hours of human review the risk promise saves on the whole archive.

The risk-control result says what share of pictures each pipeline may approve without a person,
at a promised error rate. This turns that share into hours, using how long a person actually takes,
measured in Label Studio on this archive (lead time per answer, the rater's own labelling):

  quick check     "does this picture show a scene?" -- Tasks A, A2 and H (yes/no or one scene type)
  full annotation scene label plus boxes -- the original hand-annotation project

Only answers saved before CUTOFF count, so the figure stays fixed while labelling goes on. Per
pipeline and risk target, over the 200 book-disjoint redraws (best_config_repeated.json):

  pictures to review   = (1 - approved share) x 12,110
  hours                = pictures to review x median seconds per picture / 3600 (IQR as a range)
  errors let through   = at most the risk target x 12,110 in expectation (what the promise bounds)

    python run_review_hours.py [--verify]
"""
import argparse, json, os, sqlite3
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "review_hours.json"
DB = Path(os.path.expanduser("~/.local/share/label-studio/label_studio.sqlite3"))
CUTOFF = "2026-10-07"
ARCHIVE = 12110
TASKS = {
    "quick check": ["Self-Annotation A: Negative Control Blind Spot",
                    "Self-Annotation A2: Remaining Negative Controls (750)",
                    "Task H: Second scene rating (150 pictures)"],
    "full annotation": ["Computer Vision for Historic Archives"],
}
PIPES = {"old design (disagreement triage)": ("deployed", "disagreement"),
         "corrected + no-scene score (confidence triage)": ("corrected + absence", "confidence")}
ALPHAS = ["0.05", "0.1", "0.15"]


def seconds(titles):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = ("select a.lead_time from task_completion a join task t on a.task_id=t.id join project p on t.project_id=p.id "
         "where p.title=? and a.was_cancelled=0 and a.lead_time is not null and a.created_at < ?")
    x = np.array([r[0] for t in titles for r in c.execute(q, (t, CUTOFF))], float)
    return {"answers": int(len(x)), "median_s": round(float(np.median(x)), 2),
            "p25_s": round(float(np.percentile(x, 25)), 2), "p75_s": round(float(np.percentile(x, 75)), 2)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    t = {k: seconds(v) for k, v in TASKS.items()}
    b = json.loads((SR / "best_config_repeated.json").read_text())["per_pipeline"]
    out = {"experiment": "Hours of human review saved on the whole archive by the risk-controlled auto-approval",
           "archive_pictures": ARCHIVE, "answers_before": CUTOFF, "review_time": t,
           "review_time_sources": TASKS, "baseline_review_everything_hours": {}, "per_pipeline": {}}
    for kind, s in t.items():
        out["baseline_review_everything_hours"][kind] = round(ARCHIVE * s["median_s"] / 3600, 1)
    for name, (pipe, triage) in PIPES.items():
        rows = {}
        for al in ALPHAS:
            r = b[pipe]["triage"][f"{triage}@{al}"]
            ap_ = r["auto_approved"]
            review = {k: round(ARCHIVE * (1 - ap_[k]), 0) for k in ("mean", "p2.5", "p97.5")}
            hours = {kind: {"median": round(review["mean"] * s["median_s"] / 3600, 1),
                            "iqr": [round(review["mean"] * s["p25_s"] / 3600, 1), round(review["mean"] * s["p75_s"] / 3600, 1)]}
                     for kind, s in t.items()}
            rows[al] = {"auto_approved_share": ap_, "pictures_to_review": review, "hours": hours,
                        "realised_risk": r["mean_realised_risk"],
                        "errors_let_through_expected": round(r["mean_realised_risk"] * ARCHIVE, 0),
                        "errors_let_through_bound": round(float(al) * ARCHIVE, 0)}
        out["per_pipeline"][name] = rows
    old, new = list(PIPES)
    out["saved_by_corrected_over_old"] = {
        al: {kind: round(out["per_pipeline"][old][al]["hours"][kind]["median"] - out["per_pipeline"][new][al]["hours"][kind]["median"], 1)
             for kind in t} for al in ALPHAS}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for kind, s in t.items():
        print(f"{kind}: {s['answers']} answers, median {s['median_s']} s (IQR {s['p25_s']}-{s['p75_s']}); "
              f"reviewing all {ARCHIVE:,} = {out['baseline_review_everything_hours'][kind]} h")
    for name, rows in out["per_pipeline"].items():
        print(f"\n{name}")
        for al, r in rows.items():
            print(f"  risk {al}: approve {100*r['auto_approved_share']['mean']:.1f}%, review {r['pictures_to_review']['mean']:,.0f} pictures = "
                  + ", ".join(f"{k} {v['median']} h" for k, v in r["hours"].items())
                  + f"; errors through ~{r['errors_let_through_expected']:.0f} (bound {r['errors_let_through_bound']:.0f})")
    print("\nhours saved, corrected over old:", out["saved_by_corrected_over_old"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
