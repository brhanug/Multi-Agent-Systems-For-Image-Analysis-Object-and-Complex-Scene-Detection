#!/usr/bin/env python3
"""
run_full_scale_vqa_srs.py
--------------------------
Orchestrates full-scale VQA and SRS execution on all 12,110 images.
Includes progress tracking, checkpointing, and automatic manifest integration.

Usage:
    # Start vLLM server first:
    vllm serve "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    # Then run this script:
    python scripts/core_pipeline/run_full_scale_vqa_srs.py --mode both
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check_vllm_server(url: str = "http://localhost:8000") -> bool:
    """Check if vLLM server is running."""
    import requests
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def resolve_script(*candidates: str) -> Path | None:
    """Return first existing script path relative to project root."""
    for candidate in candidates:
        candidate_path = PROJECT_ROOT / candidate
        if candidate_path.exists():
            return candidate_path
    return None

def run_vqa(dataset_root: str) -> bool:
    """Run VQA on full dataset if a known script is available."""
    print("\n" + "="*60)
    print("🧠 PHASE 1: VQA")
    print("="*60)

    vqa_script = resolve_script(
        "scripts/analysis_viz/calculate_vqa_binary_classification.py",
        "scripts/core_pipeline/calculate_vqa_binary_classification.py",
        "scripts/core_pipeline/run_llava_vqa.py",
    )
    if vqa_script is None:
        print("❌ No supported VQA script found.")
        print("   Expected one of:")
        print("   - scripts/analysis_viz/calculate_vqa_binary_classification.py")
        print("   - scripts/core_pipeline/calculate_vqa_binary_classification.py")
        print("   - scripts/core_pipeline/run_llava_vqa.py")
        return False

    cmd = ["python", str(vqa_script)]
    # Only binary-classification variants currently accept dataset_root.
    if "calculate_vqa_binary_classification.py" in vqa_script.name:
        cmd.extend([dataset_root, "--level", "both"])
    elif "run_llava_vqa.py" in vqa_script.name:
        cmd.extend(["--dataset-root", dataset_root])
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    print(f"⏱️  Estimated time: ~50 hours for 12,110 images")
    print(f"📊 Progress will be saved every 100 images\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ VQA completed successfully in {elapsed/3600:.1f} hours")
        return True
    else:
        print(f"\n❌ VQA failed with exit code {result.returncode}")
        return False

def run_srs(dataset_root: str) -> bool:
    """Run Semantic Restoration Score on full dataset."""
    print("\n" + "="*60)
    print("📊 PHASE 2: Semantic Restoration Score (SRS)")
    print("="*60)
    
    srs_script = resolve_script(
        "scripts/analysis_viz/calculate_srs_metric.py",
        "scripts/core_pipeline/calculate_srs_metric.py",
    )
    if srs_script is None:
        print("❌ No supported SRS script found.")
        return False

    cmd = ["python", str(srs_script), dataset_root]
    # Force full-scale behavior regardless of per-script defaults.
    cmd.extend(["--limit", "0"])
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    print(f"⏱️  Estimated time: ~25 hours for 12,110 images")
    print(f"📊 Progress will be saved every 100 images\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ SRS completed successfully in {elapsed/3600:.1f} hours")
        return True
    else:
        print(f"\n❌ SRS failed with exit code {result.returncode}")
        return False

def merge_to_manifest(dataset_root: str) -> bool:
    """Merge results into manifest.csv."""
    print("\n" + "="*60)
    print("💾 PHASE 3: Merging Results to Manifest")
    print("="*60)
    
    merge_script = resolve_script(
        "scripts/analysis_viz/merge_final_metadata.py",
        "scripts/core_pipeline/merge_metadata_to_manifest.py",
        "scripts/merge_metadata_to_manifest.py",
    )
    if merge_script is None:
        print("❌ No supported manifest merge script found.")
        return False

    cmd = [
        "python",
        str(merge_script),
        "--dataset-root",
        dataset_root,
        "--base-dir",
        str(PROJECT_ROOT),
    ]
    
    print(f"\n📝 Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        print("\n✅ Manifest merge completed successfully")
        return True
    else:
        print("\n❌ Manifest merge failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run full-scale VQA and SRS")
    parser.add_argument("--dataset-root", default="final_dataset", help="Dataset root directory")
    parser.add_argument("--mode", choices=["vqa", "srs", "both"], default="both",
                        help="Which analysis to run")
    parser.add_argument("--skip-server-check", action="store_true",
                        help="Skip vLLM server health check")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip manifest merge step")
    args = parser.parse_args()
    
    print("🚀 Full-Scale VQA & SRS Execution")
    print("="*60)
    print(f"Dataset: {args.dataset_root}")
    print(f"Mode: {args.mode}")
    print("="*60)
    
    # Check vLLM server
    if not args.skip_server_check:
        print("\n🔍 Checking vLLM server...")
        if not check_vllm_server():
            print("❌ vLLM server not responding at http://localhost:8000")
            print("\n💡 Start the server with:")
            print('   vllm serve "llava-hf/llava-onevision-qwen2-7b-ov-hf"')
            sys.exit(1)
        print("✅ vLLM server is running")
    
    # Run VQA
    if args.mode in ["vqa", "both"]:
        if not run_vqa(args.dataset_root):
            print("\n⚠️ VQA failed. Stopping execution.")
            sys.exit(1)
    
    # Run SRS
    if args.mode in ["srs", "both"]:
        if not run_srs(args.dataset_root):
            print("\n⚠️ SRS failed. Stopping execution.")
            sys.exit(1)
    
    # Merge to manifest
    if not args.skip_merge:
        if not merge_to_manifest(args.dataset_root):
            print("\n⚠️ Manifest merge failed.")
            sys.exit(1)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 Results:")
    print("   VQA: output depends on detected VQA script")
    print(f"   SRS: final_dataset/metadata/srs_scores.json")
    print(f"   Manifest: final_dataset/metadata/manifest.csv (updated)")
    print("\n✅ Dataset is now production-ready for thesis analysis!")

if __name__ == "__main__":
    main()
