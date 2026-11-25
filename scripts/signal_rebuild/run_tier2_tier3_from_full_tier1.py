#!/usr/bin/env python3
"""Feed the full Tier 1 re-run into Tier 2 and Tier 3, and report the whole pipeline.

Tier 1 now has genuine corpus-wide inference from all four models. This carries that
through the rest of the architecture:

  Tier 2 (downstream curation, runs on post-validation metadata only)
    Agent 2  contradiction: the VLM names an object the detector did not find
    Agent 5  grounded OCR:  Kosmos output, bucketed as in Section 6.3
    Agent 4  cross-archive linking is NOT run -- it needs image embeddings, and the
             SigLIP pass stored scene probabilities rather than embedding vectors.

  Tier 3 (frontier-only dual-LLM coordinator)
    Cannot be executed: it requires paid Gemini and Claude API keys, which this project
    does not have (Section 2.1). What IS computable, and is reported here, is the exact
    escalation load it would receive -- how many images cross the conformal threshold and
    would be routed to it.

Outputs: results/signal_rebuild/tier2_tier3_full_pipeline.json
"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
CRC = BASE / "results/multi_agent/corrected_clean_holdout_crc.json"
OUT = SR / "tier2_tier3_full_pipeline.json"

# the detector's 10-class archival taxonomy, with the surface forms a VLM actually uses
TAXONOMY = {
    "person": ["person", "people", "man", "men", "woman", "women", "figure", "adult"],
    "child":  ["child", "children", "boy", "girl", "kid", "infant", "baby"],
    "horse":  ["horse", "horses", "pony"],
    "building": ["building", "house", "church", "tower", "castle", "hut"],
    "weapon": ["weapon", "sword", "gun", "rifle", "spear"],
    "vehicle": ["vehicle", "cart", "carriage", "wagon", "car", "boat", "ship", "train"],
    "tree":   ["tree", "trees", "forest", "woods"],
    "clothing": ["clothing", "dress", "coat", "hat", "uniform", "cloak"],
    "text":   ["text", "writing", "inscription", "caption", "letters"],
    "animal": ["animal", "animals", "dog", "cat", "bird", "birds", "cow", "sheep", "deer"],
}


def read_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(rows).drop_duplicates("image_id") if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    yolo = read_jsonl([SR / "yolo_full_corpus.jsonl"])
    llava = read_jsonl(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    kosmos = read_jsonl(sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))))
    sig = read_jsonl([SR / "siglip_scene_full.jsonl"])
    sig["image_id"] = sig.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))

    d = yolo[["image_id", "n_box", "labels"]].copy()
    d["detected"] = d.labels.apply(lambda L: set(L) if isinstance(L, list) else set())
    d = d.merge(llava[["image_id", "answers"]], on="image_id", how="inner")
    d["vlm_text"] = d.answers.apply(
        lambda a: " ".join(str(x) for x in a).lower() if isinstance(a, list) else "")

    # ---- Agent 2: VLM names a taxonomy object the detector did not find ----
    def claimed(text):
        out = set()
        for cls, forms in TAXONOMY.items():
            if any(re.search(rf"\b{re.escape(f)}\b", text) for f in forms):
                out.add(cls)
        return out

    d["claimed"] = d.vlm_text.apply(claimed)
    d["contradictions"] = [sorted(c - det) for c, det in zip(d.claimed, d.detected)]
    d["n_contradictions"] = d.contradictions.apply(len)
    d["n_confirmed"] = [len(c & det) for c, det in zip(d.claimed, d.detected)]

    out = {"experiment": "Tier 1 (full re-run) fed into Tier 2 and Tier 3",
           "n_images": int(len(d)),
           "agent2_contradiction": {
               "images_with_at_least_one": int((d.n_contradictions > 0).sum()),
               "pct_of_corpus": round(100 * float((d.n_contradictions > 0).mean()), 2),
               "mean_contradictions_per_image": round(float(d.n_contradictions.mean()), 3),
               "total_flags": int(d.n_contradictions.sum()),
               "mean_confirmed_per_image": round(float(d.n_confirmed.mean()), 3),
               "most_flagged_classes": {
                   k: int(v) for k, v in
                   pd.Series([c for row in d.contradictions for c in row]).value_counts().head(6).items()},
               "note": ("A flag means the VLM named a taxonomy object the detector did not "
                        "return. Section 6.1's gold labelling of 101 such candidates found only "
                        "23.3% were genuine hallucinations; the rest were detector misses. The "
                        "rate below is therefore a candidate rate, not a hallucination rate.")}}

    # ---- Agent 5: grounded OCR, bucketed ----
    kosmos["chars"] = kosmos.kosmos_output.fillna("").str.len()

    def rep(t):
        w = str(t).split()
        return 1 - (len(set(w)) / len(w)) if w else 0.0

    kosmos["repetition"] = kosmos.kosmos_output.fillna("").map(rep)
    k = kosmos.set_index("image_id")
    out["agent5_grounded_ocr"] = {
        "n": int(len(kosmos)),
        "empty_output": int((kosmos.chars == 0).sum()),
        "candidate_real_ocr": int(((kosmos.chars >= 80) & (kosmos.repetition < 0.5)).sum()),
        "degenerate_repetitive": int((kosmos.repetition >= 0.5).sum()),
        "short_or_no_text": int((kosmos.chars < 80).sum()),
        "note": ("Section 6.3 found the candidate-real-OCR bucket only 52.5% precise on gold "
                 "labels, with median WER 100%, so bucket membership is not evidence of legible "
                 "text.")}

    out["agent4_cross_archive"] = {
        "run": False,
        "why": ("Requires image embedding vectors for nearest-neighbour search. The corpus SigLIP "
                "pass stored the five scene-category probabilities, not the embeddings, so linking "
                "cannot be recomputed without a further pass.")}

    # ---- Tier 3: the escalation load it would receive ----
    ag = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv").drop_duplicates("image_id")
    FOUR = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    ag["U_dep"] = 0.6 * ag[FOUR].std(axis=1, ddof=1) + 0.4 * (1 - ag[FOUR].mean(axis=1))
    lam = next(r for r in json.loads(CRC.read_text())["results"]
               if abs(r["target_risk_alpha"] - 0.05) < 1e-9)["conformal_threshold_lambda"]

    # Route on the REBUILT signals too. Routing on the deployed set while Tier 2 consumes the
    # rebuilt one would compare two different signal sets and the contrast would be an artifact.
    reb = pd.read_csv(SR / "full_tier1_signals.csv")
    R4 = ["A_object", "A_agree", "A_scene", "A_vlm"]
    reb["U_reb"] = 0.6 * reb[R4].std(axis=1, ddof=1) + 0.4 * (1 - reb[R4].mean(axis=1))

    m = (d[["image_id", "n_contradictions"]]
         .merge(ag[["image_id", "U_dep"]], on="image_id", how="inner")
         .merge(reb[["image_id", "U_reb"]], on="image_id", how="inner"))

    routes = {}
    for tag, col in [("deployed_signals", "U_dep"), ("rebuilt_signals", "U_reb")]:
        e = m[col] >= lam
        routes[tag] = {
            "images_escalated": int(e.sum()),
            "pct_of_corpus": round(100 * float(e.mean()), 2),
            "auto_approved": int((~e).sum()),
            "contradiction_rate_among_escalated_pct": round(
                100 * float((m.loc[e, "n_contradictions"] > 0).mean()), 2),
            "contradiction_rate_among_auto_approved_pct": round(
                100 * float((m.loc[~e, "n_contradictions"] > 0).mean()), 2),
        }
    out["tier3_escalation"] = {
        "executable": False,
        "why": ("The deployed Tier 3 pairs Gemini 2.5 Flash with Claude 3.5 Sonnet and needs paid "
                "frontier API credit, which this project does not have. Section 2.1 records it as "
                "untested rather than falsified. The load it would receive is computable and given "
                "here, under both routing signal sets."),
        "conformal_threshold": lam,
        "caveat": ("lambda was calibrated on the deployed signals. Applying it unchanged to the "
                   "rebuilt triage score is indicative only -- a genuine deployment on rebuilt "
                   "signals would recalibrate."),
        "by_routing_signal": routes,
    }
    esc = m.U_dep >= lam
    dep = out["tier3_escalation"]["by_routing_signal"]["deployed_signals"]
    rb = out["tier3_escalation"]["by_routing_signal"]["rebuilt_signals"]
    out["end_to_end_reading"] = (
        f"Of {out['n_images']} images, routing on the deployed signals clears "
        f"{dep['auto_approved']} automatically and sends {dep['images_escalated']} onward; routing "
        f"on the rebuilt signals clears {rb['auto_approved']} and sends {rb['images_escalated']}. "
        f"Tier 2 flags a VLM-detector contradiction on {out['agent2_contradiction']['pct_of_corpus']}% "
        f"of the corpus, but the flag is dominated by taxonomy mismatch rather than hallucination: "
        f"the detector recognises ten classes while the VLM describes freely, so 'clothing', 'text' "
        f"and 'tree' alone account for most flags. Under deployed routing the flag rate is HIGHER "
        f"among auto-approved images ({dep['contradiction_rate_among_auto_approved_pct']}%) than "
        f"among escalated ones ({dep['contradiction_rate_among_escalated_pct']}%), and under "
        f"rebuilt routing it is {rb['contradiction_rate_among_auto_approved_pct']}% against "
        f"{rb['contradiction_rate_among_escalated_pct']}%. Either way the contradiction signal does "
        f"not concentrate on the images the uncertainty score distrusts. That is the Architectural "
        f"Decoupling Rule behaving exactly as designed -- Tier 2 runs after routing and cannot "
        f"influence it -- but it also means Tier 2 adds no corroboration to the triage decision.")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
