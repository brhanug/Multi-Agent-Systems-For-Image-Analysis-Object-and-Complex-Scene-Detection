#!/usr/bin/env python3
"""
run_real_frontier_benchmark.py
-------------------------------
Runs ACTUAL local benchmark: DCV-MAC SAA vs. LLaVA-OneVision-7B (Monolithic Zero-Shot)
on the 100-image frontier gold subset.

Uses cached HF model: llava-hf/llava-onevision-qwen2-7b-ov-hf (offline cache)

Computes:
  - ECE (Expected Calibration Error)
  - ROC-AUC for uncertainty-driven error detection
  - Error Recall at 20% review budget

Outputs:
  results/multi_agent/real_frontier_vlm_benchmark.json
  results/multi_agent/frontier_vlm_scores_real.csv
"""

from __future__ import annotations
import json, os, gc, argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE       = Path("/data/brhanu/thesis_project")
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
VLM_CSV    = BASE / "results/multi_agent/frontier_vlm_scores.csv"
OUT_JSON   = BASE / "results/multi_agent/real_frontier_vlm_benchmark.json"
OUT_CSV    = BASE / "results/multi_agent/frontier_vlm_scores_real.csv"

MODEL_ID   = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
BUDGET_20  = 0.20
N_ECE_BINS = 10

# LLaVA-OneVision uses a chat template — do NOT use raw '<image>' prefix
QUESTION = (
    "You are evaluating a historical archive photograph for scene presence. "
    "Answer with ONLY a single float between 0.0 and 1.0 representing your "
    "confidence that this image contains a meaningful visual scene. "
    "0.0 = definitely no scene (blank/damaged/unreadable). "
    "1.0 = clearly contains a scene with people, objects, or activity. "
    "Output the number only, no explanation."
)


def ece(confs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(confs)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc  = float(labels[mask].mean())
        conf = float(confs[mask].mean())
        ece_val += (mask.sum() / total) * abs(conf - acc)
    return round(ece_val, 6)


def roc_auc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_correct = sum(float(p > n) for p in pos for n in neg)
    n_ties    = sum(float(p == n) for p in pos for n in neg) * 0.5
    return round((n_correct + n_ties) / (len(pos) * len(neg)), 6)


def recall_at_budget(uncertainty, labels, budget):
    k = max(1, int(len(uncertainty) * budget))
    idx = np.argsort(uncertainty)[::-1][:k]
    caught = labels[idx].sum()
    total  = labels.sum()
    return round(float(caught / total) if total > 0 else 1.0, 6)


def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    return p


def build_lookup():
    search_dirs = [
        BASE / "final_dataset_v1_refresh/images",
        BASE / "human_spatial_audit/images",
        BASE / "data/yolo11_v1_refresh/images",
    ]
    lookup = {}
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.rglob("*.jpg"):
                lookup[p.name] = p
                lookup[p.stem] = p
    return lookup


def find_image(image_id, lookup):
    nid = normalize_id(str(image_id))
    if nid in lookup:
        return lookup[nid]
    if f"{nid}.jpg" in lookup:
        return lookup[f"{nid}.jpg"]
    for k, v in lookup.items():
        if k.endswith(nid) or nid.endswith(k):
            return v
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print(f"Real Frontier VLM Benchmark: {MODEL_ID}")
    print("="*60)

    df_vlm = pd.read_csv(VLM_CSV)
    image_ids = df_vlm["image_id"].tolist()
    is_error  = df_vlm["is_error"].values.astype(int)
    saa_unc   = df_vlm["saa_uncertainty"].values.astype(float)

    if args.dry_run:
        image_ids = image_ids[:8]
        is_error  = is_error[:8]
        saa_unc   = saa_unc[:8]
        print(f"DRY RUN: {len(image_ids)} images")

    lookup = build_lookup()
    print(f"Image lookup: {len(lookup)} image keys mapped")

    # Load Model offline
    print(f"\nLoading {MODEL_ID} (local cache offline mode) ...")
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    print("Model loaded OK from local cache!")

    vlm_scores  = []
    valid_mask  = []
    error_count = 0

    for i, img_id in enumerate(image_ids):
        img_path = find_image(img_id, lookup)
        if img_path is None:
            print(f"  [{i+1}] SKIP {img_id}: not found")
            valid_mask.append(False)
            error_count += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            # Resize large images — historical scans can be very large
            pil_img.thumbnail((896, 896), Image.LANCZOS)

            # LLaVA-OneVision requires apply_chat_template for proper formatting
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION},
                    ],
                }
            ]
            prompt_text = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = processor(
                images=pil_img,
                text=prompt_text,
                return_tensors="pt",
            )
            # Move each tensor to the model's first device
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            eos_id = processor.tokenizer.eos_token_id
            pad_id = processor.tokenizer.pad_token_id or eos_id

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            generated = out_ids[0][inputs["input_ids"].shape[1]:]
            response  = processor.decode(generated, skip_special_tokens=True).strip()

            score = 0.5
            for token in response.split():
                try:
                    cand = float(token.strip(".,()"))
                    if 0.0 <= cand <= 1.0:
                        score = cand
                        break
                except ValueError:
                    continue

            vlm_scores.append(score)
            valid_mask.append(True)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_ids)}] score={score:.3f}  img={img_id}")

        except Exception as e:
            print(f"  [{i+1}] ERROR {img_id}: {e}")
            valid_mask.append(False)
            error_count += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if len(vlm_scores) == 0:
        print("ERROR: No valid scores. Exiting.")
        return

    valid_mask = np.array(valid_mask)
    is_error_valid = is_error[valid_mask]
    saa_unc_valid  = saa_unc[valid_mask]
    vlm_arr = np.array(vlm_scores)
    vlm_uncertainty = 0.5 - np.abs(vlm_arr - 0.5)

    # Metrics
    vlm_ece = ece(vlm_arr, 1 - is_error_valid)
    vlm_roc = roc_auc(vlm_uncertainty, is_error_valid)
    vlm_rec = recall_at_budget(vlm_uncertainty, is_error_valid, BUDGET_20)

    saa_ece  = ece(saa_unc_valid, is_error_valid)
    saa_roc  = roc_auc(saa_unc_valid, is_error_valid)
    saa_rec  = recall_at_budget(saa_unc_valid, is_error_valid, BUDGET_20)

    print(f"\n{'='*50}")
    print(f"{'Metric':<35} {'DCV-MAC (Ours)':>15} {'LLaVA-OneVision':>15}")
    print(f"{'='*50}")
    print(f"{'ECE':<35} {saa_ece:>15.4f} {vlm_ece:>15.4f}")
    print(f"{'ROC-AUC':<35} {saa_roc:>15.4f} {vlm_roc:>15.4f}")
    print(f"{'Recall@20%':<35} {saa_rec:>15.4f} {vlm_rec:>15.4f}")
    print(f"{'n valid':<35} {len(saa_unc_valid):>15} {len(vlm_scores):>15}")
    print(f"{'='*50}")

    result = {
        "metadata": {
            "model_id": MODEL_ID,
            "model_source": f"REAL — local HF cache ({MODEL_ID})",
            "n_evaluated": len(vlm_scores),
            "n_errors": int(is_error_valid.sum()),
            "n_inference_errors": error_count,
        },
        "saa_mas": {
            "ECE":             saa_ece,
            "ROC_AUC":         saa_roc,
            "Recall_at_20pct": saa_rec,
            "source": "REAL",
        },
        "monolithic_vlm": {
            "ECE":             vlm_ece,
            "ROC_AUC":         vlm_roc,
            "Recall_at_20pct": vlm_rec,
            "source": "REAL — local GPU inference, zero-shot VQA",
        },
        "latex_table_values": {
            "note": "Use these values to update Table tab:frontier_comparison in thesis",
            "VLM_ECE":     vlm_ece,
            "VLM_ROC_AUC": vlm_roc,
            "VLM_Recall20": vlm_rec,
            "SAA_ECE":      saa_ece,
            "SAA_ROC_AUC":  saa_roc,
            "SAA_Recall20": saa_rec,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    # Save CSV with VLM scores appended
    df_vlm_out = df_vlm.copy()
    vlm_col = [""] * len(df_vlm)
    j = 0
    for i2, vm in enumerate(valid_mask):
        if vm:
            vlm_col[i2] = str(vlm_scores[j])
            j += 1
    df_vlm_out["vlm_score_real"] = vlm_col
    df_vlm_out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
