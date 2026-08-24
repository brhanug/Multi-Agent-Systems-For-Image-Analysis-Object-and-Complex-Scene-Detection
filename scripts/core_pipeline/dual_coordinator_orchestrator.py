#!/usr/bin/env python3
"""
dual_coordinator_orchestrator.py
-----------------------------------
CHANGE (following the architecture correction after a full-project audit):
local mode no longer runs a same-model "Critic" pass. An earlier design
re-prompted Tier 1's own LLaVA-OneVision instance as both Primary and
Critic; a real mechanical test (run_local_dual_critic_ablation.py, 25 real
images) found the local critic never once disagreed with its own Primary
output (0/25) -- the same weights produce the same beliefs regardless of
prompt. That local self-critique step has been removed here, not just
documented as unreliable.

Local mode now runs ONLY the Primary synthesis step (a real curation
function -- reconciling sibling-agent outputs into one record still has
genuine value) and routes using Tier 1's actual, validated U_triage signal
(std of the 4 core agent scores, SCI-adjusted) against the real calibrated
threshold lambda-hat=0.5370, instead of a disconnected local heuristic mock.
Frontier mode is unchanged: Gemini proposes, Claude audits -- two
structurally independent models, the only configuration with a real (if
still unverified pending API credit) claim to a genuine second opinion.
"""
import os
import sys
import json
import argparse
import base64
import requests
import yaml
import csv
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Real, calibrated in-sample CRC threshold (exp29_conformal_risk_control.py) --
# used to route on Tier 1's actual signal when no genuine (frontier) critic ran.
TRIAGE_LAMBDA_HAT = 0.5370
SAA_WEIGHT, SCI_WEIGHT = 0.6, 0.4

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
# Same source file exp29_conformal_risk_control.py calibrates lambda-hat against.
AGENT_SCORES_PATH = os.path.join(BASE_DIR, "results/multi_agent/agent_comparison_scores.csv")
CORE_AGENT_COLS = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]


def _normalize_short_id(raw_id: str) -> str:
    """Match the normalize_id() convention used across this codebase's other
    scripts: PPN1234.../00000411_2 -> 00000411_2."""
    p = str(raw_id).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p


def load_real_triage_scores() -> dict:
    """Loads Tier 1's real, already-computed U_triage per image -- used to
    route images when no genuine (frontier) critic audit ran. Reproduces
    exp29_conformal_risk_control.py's EXACT formula (unified_U = 0.6*std(core
    agents) + 0.4*(1-mean(core agents))) over the SAME source file it
    calibrates lambda-hat=0.5370 against, so the threshold comparison here is
    actually meaningful. (Earlier draft of this function used the thesis's
    documented-but-not-actually-matching formula, 0.4*(1-SCI) from a
    separate scene-complexity file -- that produced systematically different,
    near-always-lower U values than the real calibration population; fixed
    to match the real script exactly.) Returns {short_image_id: u_triage}.

    Known, already-disclosed behavior: lambda-hat=0.5370 was calibrated on
    the curated C1 expert cohort (n=801, ~13.6% exceed it there). Verified
    against a sample of the broader, unlabeled deployment corpus, this
    threshold is crossed by close to none of it -- consistent with, not
    contradicting, the thesis's own central negative finding (Section
    clean_holdout / Table crc_clean_vs_leaky): this triage policy does not
    generalize out-of-sample. Expect local mode to route almost everything
    to semantic_index in practice; that is an honest reflection of the
    documented limitation, not a bug in this routing logic.

    SEPARATE, DEEPER ISSUE this function must defend against: the short
    image_id (e.g. "00000047_1") is only unique WITHIN one PPN publication --
    across the full corpus it collides constantly (verified: of 12,110 rows
    in AGENT_SCORES_PATH, only 1,985 short ids are unique; "00000047_1" alone
    is shared by 45+ different PPNs with genuinely different real scores).
    This appears to be a pre-existing, codebase-wide convention (the same
    normalize_id()-style short-id join is used by several other scripts in
    this project) that was never exercised at full-corpus scale before --
    every prior use was against the small, single-PPN-scoped C1/C2 expert
    cohorts (n=801/n=300), where collisions are rare or absent. Silently
    picking one of several colliding rows would be its own quiet fabrication
    (presenting an arbitrary PPN's score as if it were THIS image's score),
    so any short id that collides is deliberately EXCLUDED here rather than
    resolved by guessing -- the caller's existing "no real score available"
    path (route: hitl_queue) then applies honestly instead. A real fix
    requires the orchestrator's own image identification to carry PPN
    context end-to-end, which is out of scope for this change."""
    scores: dict[str, float] = {}
    ambiguous: set[str] = set()
    if not os.path.exists(AGENT_SCORES_PATH):
        print(f"⚠️ {AGENT_SCORES_PATH} not found -- real triage routing unavailable, will fall back to hitl_queue.")
        return scores
    with open(AGENT_SCORES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                core = [float(row[c]) for c in CORE_AGENT_COLS]
                mean = sum(core) / len(core)
                std = (sum((x - mean) ** 2 for x in core) / len(core)) ** 0.5
                u_triage = SAA_WEIGHT * std + SCI_WEIGHT * (1 - mean)
            except (KeyError, ValueError):
                continue
            short_id = _normalize_short_id(row["image_id"])
            if short_id in ambiguous:
                continue
            if short_id in scores:
                # Collision: this short id maps to >1 PPN. Don't guess which is
                # "this" image -- drop it entirely, rather than silently keep
                # whichever row happened to be read first/last.
                del scores[short_id]
                ambiguous.add(short_id)
                continue
            scores[short_id] = u_triage
    if ambiguous:
        print(f"⚠️ {len(ambiguous)} short image ids were ambiguous across multiple PPNs "
              f"and excluded from real-triage routing (will fall back to hitl_queue for those).")
    return scores


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

def process_single_image(img_id, img_entry, siglip_scenes, kosmos_txt, mode, keys, vllm_active, real_u_triage):
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
            # Frontier-only: Claude audits Gemini's Primary output -- two structurally
            # independent models. There is deliberately NO local vLLM fallback here
            # (see module docstring): re-prompting Tier 1's own LLaVA-OneVision as a
            # "critic" over its own Primary output was tested and found to add no
            # verified value (0/25 real images ever flagged a contradiction).
            if mode == "frontier" and keys.get("claude"):
                critic_response = query_frontier_critic_claude(critic_prompt, img_b64, keys.get("claude"))
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
            
        is_genuine_critic = critic_audit is not None
        if not critic_audit:
            # No genuine (frontier) critic ran for this image -- route using Tier 1's
            # actual, calibrated U_triage signal instead of a disconnected heuristic
            # mock. This is real data (std of the 4 core agent scores, SCI-adjusted)
            # checked against the real calibrated threshold, not an invented "audit".
            u_triage = real_u_triage.get(img_id)
            if u_triage is not None:
                is_ambiguous = u_triage > TRIAGE_LAMBDA_HAT
                critic_audit = {
                    "contradictions_found": None,
                    "confidence_score": round(1.0 - min(u_triage, 1.0), 4),
                    "justification": (
                        f"No frontier critic ran (mode={mode}). Routed on Tier 1's real "
                        f"U_triage={u_triage:.4f} vs. calibrated threshold={TRIAGE_LAMBDA_HAT} "
                        f"-- not a local LLM audit (that step was tested and removed)."
                    ),
                    "route": "hitl_queue" if is_ambiguous else "semantic_index"
                }
            else:
                # No real Tier 1 score available for this image either -- do not
                # invent one; default to the conservative route.
                critic_audit = {
                    "contradictions_found": None,
                    "confidence_score": None,
                    "justification": f"No frontier critic and no real U_triage score available for {img_id}; defaulting to human review.",
                    "route": "hitl_queue"
                }

        assigned_route = critic_audit.get("route", "hitl_queue")
        # The 0.70-confidence override below is specific to a genuine LLM critic's own
        # self-reported confidence scale; it must NOT re-judge the real-triage-based
        # fallback route above, which was already decided against the correct
        # calibrated threshold (TRIAGE_LAMBDA_HAT) on a different scale.
        if is_genuine_critic:
            _conf = critic_audit.get("confidence_score")
            if critic_audit.get("contradictions_found") or (_conf is not None and _conf < 0.70):
                assigned_route = "hitl_queue"

        _out_conf = critic_audit.get("confidence_score")
        return {
            "image_id": img_id,
            "image_name": img_name,
            "synthesized_metadata": primary_metadata,
            "critic_audit": critic_audit,
            "assigned_route": assigned_route,
            "confidence_score": _out_conf if _out_conf is not None else 0.5
        }
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Dual-LLM Coordinator Orchestrator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "frontier"],
                        help="Coordinator mode: local (Primary synthesis only via LLaVA-OneVision, "
                             "routed on Tier 1's real U_triage signal -- no local critic, see module "
                             "docstring) or frontier (Gemini proposes, Claude audits: two structurally "
                             "independent models, the only configuration with a real second-opinion claim)")
    parser.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument("--claude-key", type=str, default=None, help="Anthropic Claude API key")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    print("📂 Loading Tier 1's real U_triage scores (for routing when no frontier critic runs)...")
    real_u_triage = load_real_triage_scores()
    print(f"   Loaded {len(real_u_triage)} real triage scores.")

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
                vllm_active,
                real_u_triage
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


