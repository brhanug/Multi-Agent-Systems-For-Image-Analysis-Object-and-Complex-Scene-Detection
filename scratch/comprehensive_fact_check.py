#!/usr/bin/env python3
import json
import re
from pathlib import Path

# Paths
BASE = Path("/data/brhanu/thesis_project")
RESULTS_DIR = BASE / "results/multi_agent"
THESIS_TEX = BASE / "presentation/thesis_v3_updated.tex"

def load_json(name):
    p = RESULTS_DIR / name
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def main():
    # Load all json summaries
    taxonomy = load_json("taxonomy_summary.json")
    domain = load_json("domain_adaptation_summary.json")
    rq2 = load_json("rq2_disagreement_analysis.json")
    hitl = load_json("hitl_simulation_summary.json")
    retrieval = load_json("retrieval_summary.json")
    sci = load_json("scene_complexity_summary.json")
    synthetic = load_json("synthetic_human_summary.json")
    stats = load_json("statistical_report_summary.json")
    local_vlm = load_json("local_vlm_benchmark.json")

    # Read thesis content
    thesis_content = THESIS_TEX.read_text(encoding="utf-8")

    # 1. Domain Adaptation & Misalignment (RQ1)
    print_section("RQ1: Domain Gap & Misalignment")
    # Expected: detector_fail_vlm_pass_pct = 88.4 (10707 images)
    gap_pct = domain.get("domain_gap_evidence", {}).get("detector_fail_vlm_pass_pct")
    gap_count = domain.get("domain_gap_evidence", {}).get("detector_fail_vlm_pass")
    print(f"Computed Domain Gap Misalignment: {gap_pct}% ({gap_count} images)")
    
    # Check what's in the thesis
    misalign_matches = re.findall(r"([0-9\.]+)\s*\\%\s*show detector-VLM misalignment", thesis_content)
    print(f"Thesis mentions misalignment pct: {misalign_matches}")
    misalign_parentheses_matches = re.findall(r"\((student count = 0 but VLM score > 0.5|detector--VLM misalignment).*?\)", thesis_content)
    print(f"Thesis contexts: {misalign_parentheses_matches}")

    # 2. Taxonomy Coverage (RQ1)
    print_section("RQ1: Taxonomy Coverage")
    cov_pct = taxonomy.get("coverage_statistics", {}).get("pct_covered_1plus")
    cov_count = taxonomy.get("coverage_statistics", {}).get("images_with_1plus_classes")
    zipf_alpha = taxonomy.get("long_tail_zipf_exponent")
    dist_gain = taxonomy.get("distillation_efficiency", {}).get("distillation_gain")
    print(f"Computed Taxonomy Coverage: {cov_pct}% ({cov_count} images)")
    print(f"Computed Zipf Alpha: {zipf_alpha}")
    print(f"Computed Distillation Gain: {dist_gain}")

    # Check thesis
    tax_cov_thesis = re.findall(r"covers \\textbf\{([0-9\.]+)\\\%\}.*?([0-9,]+) of ([0-9,]+) images", thesis_content)
    print(f"Thesis taxonomy coverage: {tax_cov_thesis}")
    zipf_thesis = re.findall(r"Zipf \$\\alpha = ([0-9\.]+)\$", thesis_content)
    print(f"Thesis Zipf alpha: {zipf_thesis}")
    gain_thesis = re.findall(r"distillation gain of \+?([0-9\.]+)", thesis_content)
    print(f"Thesis distillation gain: {gain_thesis}")

    # 3. HITL Routing (RQ4)
    print_section("RQ4: HITL Workload Reduction")
    # Expected workload reduction
    # SAA disagreement routing reduces review budget:
    # 33.5% under random to 10% under SAA triage (70.1% workload reduction or 3.35x savings)
    print("Computed HITL parameters:")
    for k, v in hitl.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # Check thesis
    hitl_workload = re.findall(r"reduces the effective review budget from ([0-9\.]+)\\\%.*?to ([0-9\.]+)\\\%.*?a ([0-9\.]+)\\\% workload reduction", thesis_content)
    print(f"Thesis HITL workload text: {hitl_workload}")

    # 4. Complexity Routing (RQ7)
    print_section("RQ7: Complexity-Aware Curation Triage Routing")
    # 25.6% of archive to capture 96.5% of errors (3.77x lift)
    print("Computed Complexity parameters:")
    for k, v in sci.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    
    # Check thesis
    routing_thesis = re.findall(r"reviews only ([0-9\.]+)\\\%.*?to capture ([0-9\.]+)\\\% of all classification errors", thesis_content)
    print(f"Thesis Complexity routing text: {routing_thesis}")

    # 5. Local SOTA comparison (RQ3)
    print_section("RQ3: Local SOTA Benchmarking")
    print("Computed Local VLM metrics:")
    if local_vlm:
        for sys_data in local_vlm.get("local_vlm_comparison", []):
            print(f"  System: {sys_data.get('System')}")
            print(f"    ECE: {sys_data.get('Uncertainty Calibration (ECE)')}")
            print(f"    ROC-AUC: {sys_data.get('Uncertainty ROC-AUC')}")
            print(f"    Recall: {sys_data.get('Error Recall @ 20% Budget')}")

    # Check thesis Table 10 ECE/ROC-AUC/Recall
    print("Table 10 in thesis:")
    t10_lines = re.findall(r"Uncertainty Calibration \(ECE\) & .*? \\\\", thesis_content)
    print(f"  ECE rows: {t10_lines}")
    t10_auc_lines = re.findall(r"Uncertainty ROC-AUC & .*? \\\\", thesis_content)
    print(f"  ROC-AUC rows: {t10_auc_lines}")
    t10_rec_lines = re.findall(r"Error Recall @ 20\\% Budget & .*? \\\\", thesis_content)
    print(f"  Recall rows: {t10_rec_lines}")

if __name__ == "__main__":
    main()
