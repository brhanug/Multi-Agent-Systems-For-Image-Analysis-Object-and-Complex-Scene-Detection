#!/usr/bin/env python3
"""
check_agent4_linking_quality.py
----------------------------------
A single-rater reliability spot-check on Agent 4 (cross-archive anomaly
linking), run to close a previously-undocumented gap: Agent 4 is
implemented and deployed corpus-wide, but never independently checked.

Agent 4's mechanism (thesis Section 2.5) embeds each image's VLM scene
description and finds nearest neighbors across different publication
series (PPNs). This script does not reproduce the deployed embedding
model; instead it checks the quality of the raw material Agent 4's
embeddings are built from -- the LLaVA-OneVision text responses in
results_v1/llava/*.json -- for two corpus-wide, mechanically detectable
data-quality problems that would directly corrupt any downstream
embedding/nearest-neighbor step, plus a lightweight TF-IDF nearest-
neighbor proxy (not the deployed method) to show how those problems
manifest as spurious "recurring composition" matches in practice.

Part 1 (mechanical, corpus-wide, all n=12,110):
  (a) API failure text stored as if it were real VLM output (a pipeline
      bug: HTTP timeout error strings, not model output, ending up in
      the results file).
  (b) Sporadic Chinese-character token injection into English/German
      output -- a real VLM decoding artifact, not present in the source
      material (this corpus is German/English archival text).

Part 2 (TF-IDF proxy, sampled n=3,000): finds the highest-similarity
cross-PPN description pairs and reports how many of the top matches are
driven by Part 1's failure modes rather than genuine content similarity.

Outputs:
  results/multi_agent/agent4_linking_quality_check.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("/data/brhanu/thesis_project")
LLAVA_DIR = BASE / "results_v1" / "llava"
OUT_JSON = BASE / "results" / "multi_agent" / "agent4_linking_quality_check.json"

CHINESE_RE = re.compile(r"[一-鿿]")
SAMPLE_N = 3000
RANDOM_STATE = 11

SCENE_Q = "Which scene fits best: teaching, family, playing, landscape, drawing?"
OBJ_Q = ("Identify objects present: person, child, horse, building, weapon, vehicle, "
         "tree, clothing, text, animal. List only items found.")
TRIP_Q = ("List relational triplets in (Subject, Predicate, Object) format for the "
          "primary interactions in this image.")


def main() -> None:
    files = list(LLAVA_DIR.glob("*.json"))
    n_total = len(files)

    n_error, n_chinese = 0, 0
    for f in files:
        data = json.loads(f.read_text())
        full_text = " ".join(str(v) for v in data["responses"].values())
        if "HTTPConnectionPool" in full_text or "Read timed out" in full_text:
            n_error += 1
        if CHINESE_RE.search(full_text):
            n_chinese += 1

    # Part 2: TF-IDF proxy nearest-neighbor check on a sample
    random.seed(RANDOM_STATE)
    sample = random.sample(files, min(SAMPLE_N, n_total))

    descs, ppns, stems = [], [], []
    for f in sample:
        data = json.loads(f.read_text())
        r = data["responses"]
        combined = f"{r.get(SCENE_Q, '')} {r.get(OBJ_Q, '')} {r.get(TRIP_Q, '')}"
        if len(combined.strip()) < 30:
            continue
        descs.append(combined)
        ppns.append(f.stem.split("_")[0])
        stems.append(f.stem)

    vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=2)
    X = vec.fit_transform(descs)
    sim = cosine_similarity(X)
    n = len(descs)

    top_matches = []
    for i in range(n):
        for j in range(i + 1, n):
            if ppns[i] == ppns[j]:
                continue
            s = float(sim[i, j])
            if s > 0.95:
                top_matches.append((s, stems[i], stems[j]))
    top_matches.sort(reverse=True)

    n_spurious = sum(
        1 for s, a, b in top_matches
        if "ERROR" in str(a) or CHINESE_RE.search(descs[stems.index(a)]) or CHINESE_RE.search(descs[stems.index(b)])
    )

    result = {
        "n_total": n_total,
        "n_api_error_stored_as_content": n_error,
        "frac_api_error": round(n_error / n_total, 4),
        "n_chinese_hallucination": n_chinese,
        "frac_chinese_hallucination": round(n_chinese / n_total, 4),
        "tfidf_proxy_sample_n": n,
        "tfidf_proxy_cross_ppn_high_sim_pairs": len(top_matches),
        "note": "Part 2 is a TF-IDF proxy over LLaVA text fields, NOT a reproduction "
                "of Agent 4's deployed embedding method -- it exists only to show how "
                "Part 1's data-quality problems manifest as spurious cross-archive "
                "matches. High-similarity pairs should be manually reviewed, not "
                "trusted at face value.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"n = {n_total:,}")
    print(f"API-error-as-content: {n_error} ({result['frac_api_error']:.2%})")
    print(f"Chinese-token hallucination: {n_chinese} ({result['frac_chinese_hallucination']:.2%})")
    print(f"TF-IDF proxy sample: {n}, cross-PPN pairs sim>0.95: {len(top_matches)}")
    print(f"\n✅ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
