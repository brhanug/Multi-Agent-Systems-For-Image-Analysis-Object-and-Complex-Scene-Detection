#!/usr/bin/env python3
"""Pilot: can LLaVA-OneVision give per-class answers reliably? (Phase 1, C1)

The deployed four questions yield one scalar per picture. A per-class agreement
measure needs one score per class, and there are two ways to ask for it. This runs
both on 100 of the 300 human-annotated pictures, chosen rare-classes-first so every
class has positives:

  A  one yes/no question per class, scored as P(yes) from the first-token
     probabilities, in two phrasings (A1, A2) so the answer's stability can be measured;
  B  one question listing all classes, answered as a comma-separated list.

Reliability = agreement with the human boxes, agreement between the two phrasings,
and (for B) how often the reply can be parsed at all.

    CUDA_VISIBLE_DEVICES=1 python run_llava_class_prompt_pilot.py
    python run_llava_class_prompt_pilot.py --evaluate-only [--verify]
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
TRUTH = BASE / "results/signal_rebuild/class_scores/human_class_truth_300.csv"
OUT = BASE / "results/signal_rebuild/class_scores/llava_class_prompt_pilot.jsonl"
SUMMARY = BASE / "results/signal_rebuild/class_scores/llava_class_prompt_pilot_summary.json"
MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_SIDE, N_PILOT, SEED, PER_CLASS, NO_OBJECT = 640, 100, 20260916, 8, 15
NP = {"person": "a person", "child": "a child", "animal": "an animal", "building": "a building", "weapon": "a weapon",
      "vehicle": "a vehicle", "tree": "a tree", "text": "written or printed text", "furniture": "furniture", "hat": "a hat"}
CLASSES = list(NP)
A1 = "Is there {} in this picture? Answer yes or no."
A2 = "Does this picture show {}? Answer yes or no."
B = ("Which of these appear in this picture: " + ", ".join(CLASSES) +
     "? Reply with only the matching words separated by commas, or none.")


def select(t):
    rng = np.random.default_rng(SEED); chosen = []
    for c in sorted(CLASSES, key=lambda c: t[c].sum()):
        pool = [i for i in t.index[t[c] == 1] if i not in chosen]
        chosen += list(rng.permutation(pool)[:max(0, PER_CLASS - int(t.loc[chosen, c].sum()))])
    empty = [i for i in t.index[t[CLASSES].sum(axis=1) == 0] if i not in chosen]
    chosen += list(rng.permutation(empty)[:NO_OBJECT])
    rest = [i for i in t.index if i not in chosen]
    chosen += list(rng.permutation(rest)[:N_PILOT - len(chosen)])
    return t.loc[chosen]


def generate(pilot):
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(MODEL)
    chat = lambda q: proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                              add_generation_prompt=True)
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, disable_log_stats=True)
    one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    lst = SamplingParams(temperature=0.0, max_tokens=40)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for iid in pilot.index:
            img = Image.open(BASE / "final_dataset/images/restored" / f"{iid}.jpg").convert("RGB")
            if max(img.size) > MAX_SIDE:
                r = MAX_SIDE / max(img.size); img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
            reqs = [{"prompt": chat(tpl.format(NP[c])), "multi_modal_data": {"image": img}} for tpl in (A1, A2) for c in CLASSES]
            outs = llm.generate(reqs, one, use_tqdm=False)
            p_yes = []
            for o in outs:
                y = n = 0.0
                for lp in o.outputs[0].logprobs[0].values():
                    tok = (lp.decoded_token or "").strip().lower()
                    if tok == "yes": y += np.exp(lp.logprob)
                    elif tok == "no": n += np.exp(lp.logprob)
                p_yes.append(None if y + n == 0 else round(float(y / (y + n)), 5))
            reply = llm.generate([{"prompt": chat(B), "multi_modal_data": {"image": img}}], lst, use_tqdm=False)[0].outputs[0].text.strip()
            k = len(CLASSES)
            fh.write(json.dumps({"image_id": iid, "A1": dict(zip(CLASSES, p_yes[:k])), "A2": dict(zip(CLASSES, p_yes[k:])), "B_reply": reply}) + "\n")
            fh.flush()


def f1(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0


def evaluate(pilot):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    rows = {r["image_id"]: r for r in map(json.loads, open(OUT))}
    ids = [i for i in pilot.index if i in rows]; t = pilot.loc[ids]
    parsed = {i: {c for c in CLASSES if re.search(rf"\b{c}s?\b", rows[i]["B_reply"], re.I)} for i in ids}
    parse_ok = [bool(parsed[i]) or re.search(r"\bnone\b", rows[i]["B_reply"], re.I) is not None for i in ids]
    per = []
    for c in CLASSES:
        y = t[c].values
        a1 = np.array([rows[i]["A1"][c] if rows[i]["A1"][c] is not None else np.nan for i in ids])
        a2 = np.array([rows[i]["A2"][c] if rows[i]["A2"][c] is not None else np.nan for i in ids])
        ok = ~np.isnan(a1) & ~np.isnan(a2)
        b = np.array([int(c in parsed[i]) for i in ids])
        per.append({"class": c, "positives": int(y.sum()),
                    "A1_auroc": round(float(roc_auc_score(y[ok], a1[ok])), 4) if 0 < y[ok].sum() < ok.sum() else None,
                    "A1_f1": f1((a1[ok] >= 0.5).astype(int), y[ok]), "A2_f1": f1((a2[ok] >= 0.5).astype(int), y[ok]),
                    "A1_A2_same_decision": round(float(((a1[ok] >= 0.5) == (a2[ok] >= 0.5)).mean()), 4),
                    "A1_A2_spearman": round(float(spearmanr(a1[ok], a2[ok]).correlation), 4),
                    "A_unscorable": int((~ok).sum()), "B_f1": f1(b, y)})
    mac = lambda k: round(float(np.mean([p[k] for p in per if p[k] is not None])), 4)
    return {"experiment": "LLaVA per-class prompt pilot (C1)", "pictures": len(ids),
            "selection": f"rare classes first, up to {PER_CLASS} positives each, {NO_OBJECT} with no human box, rest random (seed {SEED})",
            "macro": {"A1_f1": mac("A1_f1"), "A2_f1": mac("A2_f1"), "A1_auroc": mac("A1_auroc"),
                      "A1_A2_same_decision": mac("A1_A2_same_decision"), "B_f1": mac("B_f1")},
            "B_parse_rate": round(float(np.mean(parse_ok)), 4), "per_class": per}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--evaluate-only", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    t = pd.read_csv(TRUTH, keep_default_na=False).set_index("image_id")
    pilot = select(t)
    if not a.evaluate_only: generate(pilot)
    s = evaluate(pilot)
    if a.verify:
        same = json.loads(SUMMARY.read_text()) == json.loads(json.dumps(s)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    SUMMARY.write_text(json.dumps(s, indent=2)); print(json.dumps(s["macro"]), "| B parse rate", s["B_parse_rate"])
    for p in s["per_class"]: print(p)


if __name__ == "__main__":
    raise SystemExit(main())
