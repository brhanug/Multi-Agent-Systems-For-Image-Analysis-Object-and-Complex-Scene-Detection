#!/usr/bin/env python3
"""
run_full_thesis_master_suite.py
----------------------------------
Sequentially executes every core experiment script, audits output metrics,
and validates full manuscript consistency from scratch.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

BASE_DIR = Path("/data/brhanu/thesis_project")
SCRIPTS_DIR = BASE_DIR / "scripts" / "core_pipeline"
RESULTS_DIR = BASE_DIR / "results" / "multi_agent"

SCRIPTS = [
    "run_agent_comparison.py",
    "run_multi_agent_validation.py",
    "fix_correlated_errors.py",
    "run_ppn_temporal_analysis.py",
    "run_semantic_retrieval_eval.py",
    "run_rq2_disagreement_analysis.py",
    "run_demographic_profiler.py",
    "run_geospatial_analyst.py",
    "run_6_agent_ablation.py",
    "run_research_evaluation.py",
    "run_statistical_report.py",
    "run_complexity_analysis.py",
    "run_gold_simulation.py",
    "run_hitl_simulation.py",
    "run_taxonomy_analysis.py",
    "run_domain_adaptation_analysis.py",
    "prepare_annotator_worksheets.py",
    "run_synthetic_human_baseline.py",
    "run_synthetic_worksheet_fill.py",
    "run_gold_evaluation.py",
    "run_platt_calibration.py",
    "run_detailed_viva_experiments.py",
    "run_master_consistency_audit.py",
    "run_metric_fact_check.py"
]

def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    print(f"\n=======================================================")
    print(f"▶ Running [{script_name}] ...")
    print(f"=======================================================")
    start_t = time.time()
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), capture_output=True, text=True)
    elapsed = time.time() - start_t
    if res.returncode == 0:
        print(f"✅ [{script_name}] PASSED in {elapsed:.2f}s")
        # Print last 5 lines of stdout
        lines = [l for l in res.stdout.split('\n') if l.strip()]
        for line in lines[-5:]:
            print(f"   {line}")
        return True, elapsed, res.stdout
    else:
        print(f"❌ [{script_name}] FAILED in {elapsed:.2f}s")
        print(f"--- STDOUT --- \n{res.stdout[-1000:]}")
        print(f"--- STDERR --- \n{res.stderr[-1000:]}")
        return False, elapsed, res.stderr

def main():
    print("🚀 Starting FULL Master Experiment Execution Suite...")
    summary = {}
    failed_scripts = []
    
    for script in SCRIPTS:
        success, elapsed, output = run_script(script)
        summary[script] = {
            "success": success,
            "elapsed_seconds": round(elapsed, 2)
        }
        if not success:
            failed_scripts.append(script)
            
    out_file = RESULTS_DIR / "full_master_suite_verification.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_scripts": len(SCRIPTS),
            "failed_scripts": failed_scripts,
            "summary": summary
        }, f, indent=2)
        
    print("\n=======================================================")
    if not failed_scripts:
        print("🎉 ALL 24 EXPERIMENT PIPELINE SCRIPTS PASSED SUCCESSFULLY!")
    else:
        print(f"⚠️ {len(failed_scripts)} scripts failed: {failed_scripts}")
    print(f"Report saved to: {out_file}")
    print("=======================================================")

if __name__ == "__main__":
    main()
