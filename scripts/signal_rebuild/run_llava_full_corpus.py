#!/usr/bin/env python3
"""LLaVA-OneVision VQA over the whole corpus, sharded across GPUs.

Same 4-question protocol as the deployed pipeline. One vLLM instance per GPU,
data-parallel: each shard owns a disjoint slice of the corpus and writes its own
resumable JSONL, so a killed shard restarts where it stopped.

Usage: CUDA_VISIBLE_DEVICES=<i> python run_llava_full_corpus.py --shard i --num-shards 4
"""
import argparse, glob, json, os, time
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
IMG_GLOB = str(BASE / "final_dataset/images/restored/**/*.jpg")
OUTDIR = BASE / "results/signal_rebuild/llava_full"
MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

MAX_SIDE = 640

QUESTIONS = [
    "Describe this image in one sentence.",
    "What scene or activity is shown?",
    "Are there people, children, or animals visible?",
    "Is this an indoor or outdoor scene?",
]


def image_id(p):
    p = Path(p)
    return f"{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None,
                        help="Write under this directory instead of results/signal_rebuild. "
                             "Lets a re-run be compared against the committed outputs without "
                             "overwriting them.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    if args.out_root:
        _r = Path(args.out_root)
        _r.mkdir(parents=True, exist_ok=True)
        globals()['OUTDIR'] = _r / OUTDIR.name
        print(f'writing to {OUTDIR}', flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from PIL import Image

    paths = sorted(glob.glob(IMG_GLOB, recursive=True))
    if args.limit:
        paths = paths[: args.limit]
    paths = [p for i, p in enumerate(paths) if i % args.num_shards == args.shard]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"shard_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [p for p in paths if image_id(p) not in done]
    if args.max_images:
        todo = todo[: args.max_images]
    print(f"[shard {args.shard}] own {len(paths)} | done {len(done)} | todo {len(todo)}", flush=True)
    if not todo:
        print(f"[shard {args.shard}] nothing to do", flush=True); return 0

    proc = AutoProcessor.from_pretrained(MODEL)
    prompts = [
        proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            add_generation_prompt=True)
        for q in QUESTIONS
    ]
    llm = LLM(model=MODEL, max_model_len=16384, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={"image": 1}, enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=160)

    CH = 64
    t0 = time.time(); n = 0
    with open(out_path, "a") as fh:
        for i in range(0, len(todo), CH):
            chunk = todo[i : i + CH]
            reqs, meta = [], []
            for p in chunk:
                try:
                    img = Image.open(p).convert("RGB")
                    # cap the longest side: anyres tiling otherwise emits >7k image tokens
                    if max(img.size) > MAX_SIDE:
                        r = MAX_SIDE / max(img.size)
                        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
                except Exception:
                    continue
                for qi, pr in enumerate(prompts):
                    reqs.append({"prompt": pr, "multi_modal_data": {"image": img}})
                    meta.append((image_id(p), qi))
            if not reqs:
                continue
            outs = llm.generate(reqs, sp, use_tqdm=False)
            per = {}
            for (iid, qi), o in zip(meta, outs):
                per.setdefault(iid, {})[f"q{qi}"] = o.outputs[0].text.strip()
            for iid, ans in per.items():
                answers = [ans.get(f"q{k}", "") for k in range(len(QUESTIONS))]
                substantive = sum(1 for a in answers if len(a.split()) >= 3)
                fh.write(json.dumps({
                    "image_id": iid,
                    "answers": answers,
                    "vlm_agent_raw": substantive / len(QUESTIONS),
                }) + "\n")
            fh.flush()
            n += len(per)
            el = time.time() - t0
            print(f"[shard {args.shard}] {n}/{len(todo)}  {n/el:.2f} img/s  "
                  f"eta {(len(todo)-n)/max(n/el,1e-9)/60:.1f} min", flush=True)
    el = time.time() - t0
    print(f"[shard {args.shard}] DONE {n} in {el/60:.1f} min ({n/el:.2f} img/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
