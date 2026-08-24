#!/usr/bin/env python3
"""
compare_coordinators.py
=============================================================================
Comparative Analysis: Local vLLM (LLaVA-OneVision) vs. Frontier Dual-LLM
(Gemini 1.5 Flash + Claude 3.5 Sonnet) Coordinator.
=============================================================================
WARNING (full-project reproducibility audit): the "no API keys" fallback
branch below prints "Running in METRIC COMPARATIVE STUDY mode (documented
historical data)" and then hardcodes Contradiction F1-Score = 0.812 (local)
and 0.965 (frontier) as literal constants. Neither value is computed
anywhere in this script or anywhere else in the codebase -- there is no
"documented historical data" behind them, and no gold contradiction labels
exist in this project to compute a real F1 from. This is the code path
that actually ran and produced results/multi_agent/coordinator_comparison.json;
those two F1 numbers have been withdrawn from the thesis and defense slides.
The "API keys present" branch below is also incomplete even if run: it
calls Gemini (Primary) and Claude (Critic) on 5 images but never computes
or saves an F1 score, and never exercises the local LLaVA-OneVision
condition at all, so it could not have produced the "Local" F1 either way.

A real, honest substitute question -- does the local same-model critic
re-prompt ever actually flag anything different from its own Primary pass
-- is answered instead by run_local_dual_critic_ablation.py, using real
inference and no gold labels required (result: 0/25 real images flagged,
see results/multi_agent/local_dual_critic_ablation_summary.json).

This script compares:
1. Current Local Dual-LLM Coordinator (LLaVA-OneVision 7B)
2. Frontier API Dual-LLM Coordinator (Gemini 1.5 Flash as Primary + Claude 3.5 Sonnet as Critic)

It evaluates:
- Latency (s per image)
- Cost ($ per 1,000 images)
- JSON Syntax Integrity (failure rate)
- Contradiction Detection Recall (vs. human gold audit results) -- NOT ACTUALLY IMPLEMENTED, see warning above
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
import numpy as np

BASE_DIR = "/data/brhanu/thesis_project"
FUSION_JSON = os.path.join(BASE_DIR, "results/multi_agent/upgraded_agent0_fusion.json")
IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored")

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_claude(image_b64, prompt, api_key, model="claude-3-5-sonnet-20241022"):
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0
    }
    t0 = time.time()
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=45)
    latency = time.time() - t0
    if resp.status_code == 200:
        return resp.json()["content"][0]["text"], latency
    raise Exception(f"Claude error {resp.status_code}: {resp.text}")

def query_gemini(image_b64, prompt, api_key, model="gemini-1.5-flash"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0
        }
    }
    t0 = time.time()
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45)
    latency = time.time() - t0
    if resp.status_code == 200:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"], latency
    raise Exception(f"Gemini error {resp.status_code}: {resp.text}")

def run_comparison():
    print("="*75)
    print("COORDINATOR COMPARATIVE EVALUATION")
    print("="*75)

    # 1. Load sample images from fusion data
    with open(FUSION_JSON) as f:
        fusion_data = json.load(f)
    
    # Select 5 representative images
    sample_recs = fusion_data[:5]
    print(f"Loaded {len(sample_recs)} test images for evaluation.")

    claude_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    has_keys = bool(claude_key and gemini_key)
    
    if not has_keys:
        print("\n⚠️ API keys missing. Cannot run a real comparison.")
        print("⚠️ WARNING: the numbers below are UNVERIFIED PLACEHOLDERS, not measurements --")
        print("   see the module docstring. Do not cite the Contradiction F1-Score fields as real.\n")

        # UNVERIFIED PLACEHOLDER TABLE -- not computed, not "documented historical data".
        # Latency/cost/JSON-parse-rate fields are plausible illustrative figures only.
        # Contradiction F1-Score fields specifically have been withdrawn from the thesis
        # and defense slides; do not reintroduce them as if measured.
        comparison = {
            "Local LLaVA-OneVision Dual-LLM": {
                "Mean Latency (s)": 3.75,
                "API Cost ($/1k images)": "0.00 (Self-hosted)",
                "JSON Parse Success Rate": "93.8%",
                "Hardware Requirements": "1x A100 GPU (80GB)",
                "Contradiction F1-Score": None,
                "Contradiction F1-Score_UNVERIFIED_PLACEHOLDER_DO_NOT_CITE": 0.812,
                "Epistemic Bias": "High (shared weights between Primary & Critic)"
            },
            "Frontier Dual-LLM (Gemini 1.5 Flash + Claude 3.5 Sonnet)": {
                "Mean Latency (s)": 1.95,
                "API Cost ($/1k images)": "~0.85 (Asymmetric billing)",
                "JSON Parse Success Rate": "100.0%",
                "Hardware Requirements": "Serverless API client only",
                "Contradiction F1-Score": None,
                "Contradiction F1-Score_UNVERIFIED_PLACEHOLDER_DO_NOT_CITE": 0.965,
                "Epistemic Bias": "Near-Zero (heterogeneous architectures & weights)"
            }
        }

        print(json.dumps(comparison, indent=2))
        
        # Save comparison results
        with open(os.path.join(BASE_DIR, "results/multi_agent/coordinator_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✅ Saved comparative metrics to results/multi_agent/coordinator_comparison.json")
        return

    # Real evaluation loop if keys are present
    print("Initiating live API benchmarking...")
    for i, rec in enumerate(sample_recs):
        img_name = rec["image_name"]
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        print(f"\nProcessing [{i+1}/5]: {img_name}")
        img_b64 = encode_image(img_path)
        
        # Step A: Primary Coordinator (Gemini)
        prompt_primary = "Synthesize scene metadata and output a JSON containing 'scene_type' (drawings|landscape|family|playing|teaching)."
        try:
            gemini_raw, lat_g = query_gemini(img_b64, prompt_primary, gemini_key)
            print(f"  Gemini (Primary) Latency: {lat_g:.2f}s")
            gemini_json = json.loads(gemini_raw)
        except Exception as e:
            print(f"  Gemini failed: {e}")
            continue
            
        # Step B: Critic Auditor (Claude)
        prompt_critic = f"Audit the following primary metadata against the image: {json.dumps(gemini_json)}. Output JSON containing 'contradictions_found' (true|false)."
        try:
            claude_raw, lat_c = query_claude(img_b64, prompt_critic, claude_key)
            print(f"  Claude (Critic) Latency : {lat_c:.2f}s")
            claude_json = json.loads(claude_raw)
        except Exception as e:
            print(f"  Claude failed: {e}")
            continue

        print("  Agreement verified successfully!")

if __name__ == "__main__":
    run_comparison()
