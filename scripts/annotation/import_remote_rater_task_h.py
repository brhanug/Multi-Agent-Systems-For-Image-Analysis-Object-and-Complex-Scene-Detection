#!/usr/bin/env python3
"""Read a remote rater's Label Studio export for Task H and measure agreement with the first rater.

The second rater works in their own Label Studio on their own computer (the package in
deliverables/rater2_task_h), so their export names each picture by the file they uploaded, which
Label Studio prefixes with a short hash: `/data/upload/1/3f9a2c1b-taskh_<image_id>.jpg`. This strips
that back to the image id, takes each picture's submitted scene choice, and compares it with the
first rater's scene label for the same picture (human_spatial_audit/user_annotations_300.csv, blank
= none).

Reported: how many pictures were rated, overall Cohen's kappa with a bootstrap interval, and per
category the raw agreement, kappa, both raters' positive counts and PABAK -- the same measures as
the existing per-category kappa on the 124 double-reviewed pictures, so the two can be pooled or
compared. The rater's raw answers are kept beside the result.

    python import_remote_rater_task_h.py <export.json> --rater "<name>" [--out <dir>]
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path("/data/brhanu/thesis_project")
FIRST = BASE / "human_spatial_audit/user_annotations_300.csv"
EXPECTED = BASE / "deliverables/rater2_task_h/file_list.txt"
OUT_DIR = BASE / "results/multi_agent"
CATS = ["teaching", "family", "playing", "landscape", "drawing", "none"]
N_BOOT, SEED = 2000, 20260930


def image_id_of(path):
    name = Path(str(path).split("?")[-1]).name
    name = re.sub(r"^[0-9a-f]{8}-", "", name)          # Label Studio's upload prefix
    name = re.sub(r"^taskh_", "", name)
    return re.sub(r"\.(jpe?g|png)$", "", name, flags=re.I)


def read_export(path):
    rows = []
    for task in json.loads(Path(path).read_text()):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = sorted(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        pick = {r["from_name"]: r["value"]["choices"][0] for r in ann.get("result", [])
                if r.get("type") == "choices" and r.get("value", {}).get("choices")}
        if "scene" in pick:
            rows.append({"image_id": image_id_of(task["data"].get("image", "")),
                         "scene_rater2": pick["scene"].strip().lower(),
                         "confidence_rater2": pick.get("confidence", "")})
    d = pd.DataFrame(rows)
    dup = d.image_id.duplicated().sum()
    return d.drop_duplicates("image_id", keep="last"), int(dup)


def kappa(a, b):
    return None if len(set(a)) < 2 and len(set(b)) < 2 else float(cohen_kappa_score(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export"); ap.add_argument("--rater", required=True); ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    d, dup = read_export(a.export)
    expected = {re.sub(r"\.jpe?g$", "", l.strip().replace("taskh_", "")) for l in open(EXPECTED) if l.strip()}
    unknown = sorted(set(d.image_id) - expected)
    first = pd.read_csv(FIRST, keep_default_na=False)[["image_id", "scene"]]
    first["scene_rater1"] = first.scene.replace("", "none").str.strip().str.lower()
    m = d[d.image_id.isin(expected)].merge(first[["image_id", "scene_rater1"]], on="image_id")
    r1, r2 = m.scene_rater1.values, m.scene_rater2.values

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(m), len(m)) for _ in range(N_BOOT)]
    labels = sorted(set(r1) | set(r2))
    boot = [cohen_kappa_score(r1[i], r2[i], labels=labels) for i in idx]
    per = {}
    for c in CATS:
        x, y = (r1 == c).astype(int), (r2 == c).astype(int)
        po = float((x == y).mean()); k = kappa(x, y)
        bs = [v for v in (kappa(x[i], y[i]) for i in idx) if v is not None]
        per[c] = {"rater1_positives": int(x.sum()), "rater2_positives": int(y.sum()),
                  "raw_agreement": round(po, 4), "cohens_kappa": None if k is None else round(k, 4),
                  "kappa_ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)] if bs else None,
                  "pabak": round(2 * po - 1, 4)}
    out = {"experiment": "Task H: second scene rating by an independent remote rater",
           "second_rater": a.rater, "export_file": Path(a.export).name,
           "pictures_expected": len(expected), "pictures_rated": int(len(m)),
           "not_rated": sorted(expected - set(m.image_id)), "unexpected_files": unknown, "duplicates_dropped": dup,
           "overall": {"cohens_kappa": round(float(cohen_kappa_score(r1, r2, labels=labels)), 4),
                       "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                       "raw_agreement": round(float((r1 == r2).mean()), 4)},
           "per_category": per,
           "confusion_rater1_rows_rater2_cols": pd.crosstab(m.scene_rater1, m.scene_rater2).to_dict()}
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    (od / "task_h_inter_rater.json").write_text(json.dumps(out, indent=2))
    m.to_csv(od / "task_h_second_rater_labels.csv", index=False)
    print(f"{out['pictures_rated']} of {out['pictures_expected']} pictures rated by {a.rater}"
          + (f"; {len(out['not_rated'])} missing" if out["not_rated"] else "")
          + (f"; {len(unknown)} unexpected files ignored" if unknown else ""))
    o = out["overall"]
    print(f"overall Cohen's kappa {o['cohens_kappa']:.3f}  95% CI [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]  raw agreement {o['raw_agreement']:.1%}")
    for c, r in per.items():
        k = "  n/a " if r["cohens_kappa"] is None else f"{r['cohens_kappa']:+.3f}"
        print(f"  {c:10s} kappa {k}  positives {r['rater1_positives']:3d} / {r['rater2_positives']:3d}  PABAK {r['pabak']:+.3f}")


if __name__ == "__main__":
    main()
