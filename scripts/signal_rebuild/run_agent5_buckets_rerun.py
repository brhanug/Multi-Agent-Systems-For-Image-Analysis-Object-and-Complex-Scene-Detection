#!/usr/bin/env python3
"""Agent 5's three-bucket rule (check_agent5_ocr_degeneracy.py: unique-word ratio < 0.3 = repetition; a
caption opener = hallucinated captioning; the rest = candidate OCR) applied unchanged to the Kosmos-2.5
outputs of the full rerun (results/signal_rebuild/kosmos_full), beside the deployed outputs' figures.

    python run_agent5_buckets_rerun.py [--verify]
"""
import argparse, glob, json, re

import run_best_config_repeated as rb

OUT = rb.SR / "agent5_buckets_rerun.json"
CAPTION_RE = re.compile(r'!\[(the image (above|below)?|a picture of|the picture|an image of)', re.IGNORECASE)


def ratio(t):
    w = t.split(); return 1.0 if len(w) < 5 else len(set(w)) / len(w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    rows = {}
    for f in sorted(glob.glob(str(rb.SR / "kosmos_full/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); rows[r["image_id"]] = r.get("kosmos_output") or ""
    n = len(rows); d = c = 0
    for t in rows.values():
        if ratio(t) < 0.3: d += 1
        elif CAPTION_RE.search(t[:120]): c += 1
    out = {"n": n, "degenerate_repetition": round(d / n, 4), "hallucinated_captioning_min": round(c / n, 4),
           "candidate_real_ocr": round((n - d - c) / n, 4)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
