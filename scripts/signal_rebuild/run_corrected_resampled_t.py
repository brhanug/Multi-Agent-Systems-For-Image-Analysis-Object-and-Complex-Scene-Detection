#!/usr/bin/env python3
"""Every paired difference over the 200 redraws, read with the corrected resampled t-test of Nadeau and
Bengio (2003), beside the percentile range the thesis reports.

The redraws share most of their pictures, so the plain standard error of the mean, s / sqrt(J), is far too
small. Nadeau and Bengio correct the variance of the mean to (1/J + n_test/n_train) s^2. This reruns the
experiment scripts unchanged (their outputs go to a temporary folder, the stored results are not touched),
records each per-redraw difference as it is summarised, and reports for each: the mean, the 2.5/97.5
percentiles of single redraws (what the thesis reports), and the corrected 95% interval for the mean.

    python run_corrected_resampled_t.py [--verify]
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np
from scipy import stats

import run_best_config_repeated as rb

OUT = rb.SR / "corrected_resampled_t.json"
SCRIPTS = ["run_best_config_repeated", "run_matched_risk_comparison", "run_ablation_corrected", "run_scene_classes",
           "run_combination_weights", "run_scene_classes_joint", "run_combination_selection", "run_restoration_ablation"]
RATIO_POOL, RATIO_OBJECTS = 0.2 / 0.6, 0.3 / 0.4      # test / training share of the books


def paths_with_paired(d, path=()):
    if isinstance(d, dict):
        if "share_of_draws_above_zero" in d:
            yield path, d
        for k, v in d.items():
            yield from paths_with_paired(v, path + (k,))


def corrected(xs, ratio):
    xs = np.asarray(xs, float); J = len(xs); m = xs.mean(); s2 = xs.var(ddof=1)
    se = np.sqrt((1.0 / J + ratio) * s2); t = stats.t.ppf(0.975, J - 1)
    return {"mean": round(float(m), 4), "p2.5": round(float(np.percentile(xs, 2.5)), 4), "p97.5": round(float(np.percentile(xs, 97.5)), 4),
            "corrected_low": round(float(m - t * se), 4), "corrected_high": round(float(m + t * se), 4),
            "percentile_range_excludes_zero": bool(np.percentile(xs, 2.5) > 0 or np.percentile(xs, 97.5) < 0),
            "corrected_excludes_zero": bool(m - t * se > 0 or m + t * se < 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp()); out = {"method": "Nadeau and Bengio (2003) corrected resampled t, 95%, J = 200",
                                           "ratio_pool": round(RATIO_POOL, 4), "ratio_objects": round(RATIO_OBJECTS, 4), "results": {}}
    orig = rb.paired
    for name in SCRIPTS:
        captured = []
        rb.paired = lambda xs, _c=captured: (_c.append(np.asarray(xs, float)), orig(xs))[1]
        mod = __import__(name)
        if hasattr(mod, "paired") and mod is not rb:
            mod.paired = rb.paired
        mod.OUT = tmp / f"{name}.json"; sys.argv = [name]
        mod.main()
        res = json.loads(mod.OUT.read_text())
        found = list(paths_with_paired(res))
        assert len(found) == len(captured), (name, len(found), len(captured))
        for (p, d), xs in zip(found, captured):
            assert abs(d["mean"] - round(float(xs.mean()), 4)) < 1e-9, (name, p)
            ratio = RATIO_OBJECTS if p and p[0] == "object_classes" else RATIO_POOL
            out["results"]["|".join((name,) + p)] = corrected(xs, ratio)
    rb.paired = orig
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    flips = {k: v for k, v in out["results"].items() if v["percentile_range_excludes_zero"] != v["corrected_excludes_zero"]}
    print(f"{len(out['results'])} differences; {len(flips)} read differently by the two rules:")
    for k, v in flips.items(): print(" ", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
