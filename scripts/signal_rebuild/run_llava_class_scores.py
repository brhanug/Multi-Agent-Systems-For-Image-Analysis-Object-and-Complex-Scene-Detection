#!/usr/bin/env python3
"""Per-class LLaVA-OneVision scores over the whole corpus (Phase 1, C1).

The pilot (run_llava_class_prompt_pilot.py) found one yes/no question per class, read
as P(yes) from the first-token probabilities, the reliable way to get per-class answers:
macro AUROC 0.91 on 100 human-annotated pictures, and two phrasings agreeing on 95.7% of
decisions. This asks that question for every class in CLASSES, which covers both the
detector's names and the rater's, so the scores serve whichever class list is frozen.

Sharded and resumable, like run_llava_full_corpus.py.

    CUDA_VISIBLE_DEVICES=<gpu> python run_llava_class_scores.py --shard i --num-shards n
"""
import argparse, glob, json, time
from pathlib import Path

import numpy as np

BASE = Path("/data/brhanu/thesis_project")
IMG_GLOB = str(BASE / "final_dataset/images/restored/**/*.jpg")
OUTDIR = BASE / "results/signal_rebuild/class_scores/llava_class_scores"
MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_SIDE, CHUNK = 640, 32
NP = {"person": "a person", "child": "a child", "animal": "an animal", "horse": "a horse", "building": "a building",
      "weapon": "a weapon", "vehicle": "a vehicle", "tree": "a tree", "text": "written or printed text",
      "clothing": "clothing", "furniture": "furniture", "hat": "a hat"}
QUESTION = "Does this picture show {}? Answer yes or no."   # phrasing A2 of the pilot
CLASSES = list(NP)


def image_id(p):
    p = Path(p)
    return f"{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from PIL import Image
    paths = sorted(glob.glob(IMG_GLOB, recursive=True))[: a.limit or None]
    paths = [p for i, p in enumerate(paths) if i % a.num_shards == a.shard]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"shard_{a.shard}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try: done.add(json.loads(line)["image_id"])
            except Exception: pass
    todo = [p for p in paths if image_id(p) not in done]
    print(f"[shard {a.shard}] own {len(paths)} | done {len(done)} | todo {len(todo)}", flush=True)
    if not todo: return 0
    proc = AutoProcessor.from_pretrained(MODEL)
    prompts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(v)}]}],
                                        add_generation_prompt=True) for v in NP.values()]
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, enable_prefix_caching=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for i in range(0, len(todo), CHUNK):
            reqs, meta = [], []
            for p in todo[i:i + CHUNK]:
                try:
                    img = Image.open(p).convert("RGB")
                    if max(img.size) > MAX_SIDE:
                        r = MAX_SIDE / max(img.size); img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
                except Exception:
                    continue
                for c, pr in zip(NP, prompts):
                    reqs.append({"prompt": pr, "multi_modal_data": {"image": img}}); meta.append((image_id(p), c))
            # one question per picture first: its picture tokens enter the prefix cache, and the
            # remaining questions reuse them instead of each re-encoding the same picture
            first = [k for k, (_, c) in enumerate(meta) if c == CLASSES[0]]
            taken = set(first); rest = [k for k in range(len(meta)) if k not in taken]
            outs = [None] * len(meta)
            for group in (first, rest):
                for k, o in zip(group, llm.generate([reqs[k] for k in group], sp, use_tqdm=False)):
                    outs[k] = o
            per = {}
            for (iid, c), o in zip(meta, outs):
                y = no = 0.0
                for lp in o.outputs[0].logprobs[0].values():
                    tok = (lp.decoded_token or "").strip().lower()
                    if tok == "yes": y += np.exp(lp.logprob)
                    elif tok == "no": no += np.exp(lp.logprob)
                per.setdefault(iid, {})[c] = None if y + no == 0 else round(float(y / (y + no)), 5)
            for iid, s in per.items():
                fh.write(json.dumps({"image_id": iid, "p_yes": s}) + "\n")
            fh.flush(); n += len(per); el = time.time() - t0
            if (i // CHUNK) % 10 == 0:
                print(f"[shard {a.shard}] {n}/{len(todo)}  {n/el:.2f} img/s  eta {(len(todo)-n)/max(n/el,1e-9)/60:.0f} min", flush=True)
    print(f"[shard {a.shard}] DONE {n} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
