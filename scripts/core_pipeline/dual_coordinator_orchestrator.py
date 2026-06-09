#!/usr/bin/env python3
import os
import sys
import json
import argparse
import base64
import requests
import yaml
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# CONFIGURATION
# ==============================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CFG = load_config()
BASE_DIR = CFG['project']['base_dir']
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored")
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) == 0:
    IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/restored")

YOLO_RESULTS_PATH = os.path.join(BASE_DIR, "results/yolo11_seg_sam_results.json")
KOSMOS_RESULTS_PATH = os.path.join(BASE_DIR, "results/kosmos_grounding.jsonl")
SIGLIP_RESULTS_PATH = os.path.join(BASE_DIR, "results/scene_labels/scene_labels_siglip.json")

OUTPUT_JSON = os.path.join(BASE_DIR, "results/multi_agent/upgraded_agent0_fusion.json")

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_vllm_with_image(prompt, image_base64):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(VLLM_URL, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"⚠️ vLLM server returned status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Error querying vLLM: {e}")
        return None

def query_frontier_primary_gemini(prompt, image_base64, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return None
    except Exception:
        return None

def query_frontier_critic_claude(prompt, image_base64, api_key):
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0
    }
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        return None
    except Exception:
        return None

def process_single_image(img_id, img_entry, siglip_scenes, kosmos_txt, mode, keys, vllm_active):
    try:
        img_name = img_entry["image_name"]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            return None
            
        yolo_dets = img_entry.get("detections", [])
        yolo_dets_clean = []
        for d in sorted(yolo_dets, key=lambda x: x.get("confidence", 0.0), reverse=True)[:10]:
            yolo_dets_clean.append({
                "class_name": d.get("class_name"),
                "confidence": d.get("confidence"),
                "bbox": d.get("bbox")
            })
            
        kosmos_txt_clean = kosmos_txt[:500]
        
        # Primary VLM Prompt
        primary_prompt = f"""You are the Primary Coordinator in a Multi-Agent system for historical image analysis.
Your task is to synthesize the following detection signals, OCR text, and scene predictions into a final set of metadata:
1. YOLOv11-seg + SAM detections: {json.dumps(yolo_dets_clean)}
2. Kosmos-2.5 Markdown text: {kosmos_txt_clean}
3. SigLIP scene classifications: {json.dumps(siglip_scenes)}

Analyze the image and the provided metadata, resolve contradictions, and output a structured JSON ONLY. Do not output conversational text outside JSON. Keep lists concise.
JSON format:
{{
  "objects": [
     {{"name": "object name", "confidence": 0.9, "bbox": [x1, y1, x2, y2]}}
  ],
  "scene_type": "one of drawings, landscapes, family, playing, teaching",
  "ocr_transcription": "final transcription of any text in the image",
  "reasoning": "brief justification"
}}"""

        primary_metadata = None
        critic_audit = None
        
        img_b64 = encode_image_base64(img_path)
        
        # 1. Primary Query
        if mode == "frontier" and keys.get("gemini"):
            primary_response = query_frontier_primary_gemini(primary_prompt, img_b64, keys.get("gemini"))
        elif vllm_active:
            primary_response = query_vllm_with_image(primary_prompt, img_b64)
        else:
            primary_response = None

        if primary_response:
            try:
                cleaned_resp = primary_response.strip()
                if cleaned_resp.startswith("```json"):
                    cleaned_resp = cleaned_resp[7:]
                if cleaned_resp.endswith("```"):
                    cleaned_resp = cleaned_resp[:-3]
                primary_metadata = json.loads(cleaned_resp.strip())
            except Exception:
                pass
        
        # 2. Critic Query
        if primary_metadata:
            critic_prompt = f"""You are the Critic Auditor in a Multi-Agent system.
Your task is to audit the synthesized metadata generated by the Primary Coordinator against the image.
Synthesized Metadata: {json.dumps(primary_metadata)}

Perform the following checks:
1. Contradictions: Does the metadata claim something not present in the image, or miss something obvious?
2. Realism assessment: Are the labels and locations plausible?

Output a JSON ONLY. Do not output conversational text outside JSON.
JSON format:
{{
  "contradictions_found": true/false,
  "confidence_score": 0.85,
  "justification": "explanation of your audit",
  "route": "semantic_index" or "hitl_queue"
}}"""
            if mode == "frontier" and keys.get("claude"):
                critic_response = query_frontier_critic_claude(critic_prompt, img_b64, keys.get("claude"))
            elif vllm_active:
                critic_response = query_vllm_with_image(critic_prompt, img_b64)
            else:
                critic_response = None

            if critic_response:
                try:
                    cleaned_resp = critic_response.strip()
                    if cleaned_resp.startswith("```json"):
                        cleaned_resp = cleaned_resp[7:]
                    if cleaned_resp.endswith("```"):
                        cleaned_resp = cleaned_resp[:-3]
                    critic_audit = json.loads(cleaned_resp.strip())
                except Exception:
                    pass

        # Fallback / Mock values if VLM fails or is inactive
        if not primary_metadata:
            scene_type = siglip_scenes[0]["label"] if siglip_scenes else "drawings"
            primary_metadata = {
                "objects": [{"name": d["class_name"], "confidence": d["confidence"], "bbox": d["bbox"]} for d in yolo_dets],
                "scene_type": scene_type,
                "ocr_transcription": kosmos_txt[:100] if kosmos_txt else "",
                "reasoning": f"Generated via rule-based fallback since {mode} VLM was unavailable."
            }
            
        if not critic_audit:
            has_contradiction = len(yolo_dets) == 0 and len(kosmos_txt) > 50
            confidence = 0.90 if len(yolo_dets) > 0 else 0.50
            critic_audit = {
                "contradictions_found": has_contradiction,
                "confidence_score": confidence,
                "justification": f"Heuristic validation check ({mode} offline).",
                "route": "semantic_index" if (confidence >= 0.70 and not has_contradiction) else "hitl_queue"
            }

        assigned_route = critic_audit.get("route", "hitl_queue")
        if critic_audit.get("contradictions_found", False) or critic_audit.get("confidence_score", 0.0) < 0.70:
            assigned_route = "hitl_queue"

        return {
            "image_id": img_id,
            "image_name": img_name,
            "synthesized_metadata": primary_metadata,
            "critic_audit": critic_audit,
            "assigned_route": assigned_route,
            "confidence_score": critic_audit.get("confidence_score", 0.5)
        }
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Dual-LLM Coordinator Orchestrator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "frontier"],
                        help="Coordinator mode: local (LLaVA-OneVision) or frontier (Gemini + Claude)")
    parser.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument("--claude-key", type=str, default=None, help="Anthropic Claude API key")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 1. Load inputs
    print("📂 Loading detection and scene labeling outputs...")
    if not os.path.exists(YOLO_RESULTS_PATH):
        print(f"❌ YOLO results not found at {YOLO_RESULTS_PATH}.")
        sys.exit(1)
        
    with open(YOLO_RESULTS_PATH, "r") as f:
        yolo_data = json.load(f)
        
    if not os.path.exists(SIGLIP_RESULTS_PATH):
        print(f"❌ SigLIP results not found at {SIGLIP_RESULTS_PATH}.")
        sys.exit(1)
        
    with open(SIGLIP_RESULTS_PATH, "r") as f:
        siglip_data = json.load(f)

    kosmos_data = {}
    if os.path.exists(KOSMOS_RESULTS_PATH):
        with open(KOSMOS_RESULTS_PATH, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    img_id = os.path.splitext(os.path.basename(item["image"]))[0]
                    kosmos_data[img_id] = item["kosmos_output"]

    existing_results = []
    processed_ids = set()
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                existing_results = json.load(f)
                processed_ids = {entry["image_id"] for entry in existing_results if "image_id" in entry}
            print(f"🔄 Found {len(processed_ids)} already processed images in {OUTPUT_JSON}.")
        except Exception as e:
            print(f"⚠️ Failed to load existing output JSON: {e}.")

    all_images = sorted(list(yolo_data.keys()))
    images_to_process = [img_id for img_id in all_images if img_id not in processed_ids]
    
    if args.limit:
        images_to_process = images_to_process[:args.limit]
        print(f"⚠️ Limiting to {args.limit} images for testing.")

    if not images_to_process:
        print("✅ All images are already processed! Nothing to do.")
        return

    # Check connection / credentials based on mode
    keys = {
        "gemini": args.gemini_key or os.environ.get("GEMINI_API_KEY"),
        "claude": args.claude_key or os.environ.get("ANTHROPIC_API_KEY")
    }

    # Verify vLLM availability as fallback
    print(f"📡 Testing connection to vLLM server at {VLLM_URL}...")
    try:
        test_resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        vllm_active = test_resp.status_code == 200
        print("✅ vLLM server is active and reachable!")
    except Exception:
        print("⚠️ vLLM server is NOT reachable. Fallbacks will use rule-based metrics.")
        vllm_active = False

    if args.mode == "frontier":
        if keys.get("gemini"):
            print("✅ Google Gemini API key configured for Primary Coordinator.")
        else:
            print("⚠️ Google Gemini API key missing. Primary will fall back to local vLLM/mock.")
        if keys.get("claude"):
            print("✅ Anthropic Claude API key configured for Critic Auditor.")
        else:
            print("⚠️ Anthropic Claude API key missing. Critic will fall back to local vLLM/mock.")

    results = list(existing_results)

    print(f"🚀 Submitting {len(images_to_process)} tasks (mode={args.mode}) to ThreadPoolExecutor with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_img = {}
        for img_id in images_to_process:
            img_entry = yolo_data[img_id]
            siglip_scenes = siglip_data.get(img_id, [])
            kosmos_txt = kosmos_data.get(img_id, "")
            
            future = executor.submit(
                process_single_image,
                img_id,
                img_entry,
                siglip_scenes,
                kosmos_txt,
                args.mode,
                keys,
                vllm_active
            )
            future_to_img[future] = img_id
            
        for future in tqdm(as_completed(future_to_img), total=len(images_to_process)):
            res = future.result()
            if res is not None:
                results.append(res)
                try:
                    with open(OUTPUT_JSON, "w") as f:
                        json.dump(results, f, indent=4)
                except Exception as e:
                    print(f"⚠️ Error saving progress to {OUTPUT_JSON}: {e}")

    print(f"✅ Coordinator processing completed. Final outputs saved to {OUTPUT_JSON}")
    
    routes = [r["assigned_route"] for r in results]
    semantic_index_count = routes.count("semantic_index")
    hitl_queue_count = routes.count("hitl_queue")
    print(f"📊 Summary: Sent {semantic_index_count} to semantic_index, {hitl_queue_count} to hitl_queue.")

if __name__ == "__main__":
    main()


