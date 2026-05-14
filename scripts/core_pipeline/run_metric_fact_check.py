#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
RESULTS_DIR = Path("results/multi_agent")
OUTPUT_FILE = RESULTS_DIR / "fact_check_report.json"

def load_json(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def fact_check():
    checks = {}

    # 1. RQ1: Domain Adaptation & Taxonomy
    taxonomy = load_json("taxonomy_summary.json")
    if taxonomy:
        checks["RQ1_taxonomy_coverage"] = taxonomy.get("coverage_statistics", {}).get("pct_covered_1plus")
        checks["RQ1_distillation_gain"] = taxonomy.get("distillation_efficiency", {}).get("distillation_gain")

    domain = load_json("domain_adaptation_summary.json")
    if domain:
        checks["RQ1_detector_vlm_gap"] = domain.get("domain_gap_evidence", {}).get("detector_fail_vlm_pass_pct")

    # 2. RQ2: Disagreement Analysis
    rq2 = load_json("rq2_disagreement_analysis.json")
    if rq2:
        checks["RQ2_disagreement_issue_correlation"] = rq2.get("disagreement_issue_correlation")
        checks["RQ2_best_roc_auc"] = rq2.get("best_roc_auc")

    # 3. RQ4: HITL Efficiency
    hitl = load_json("hitl_simulation_summary.json")
    if hitl:
        checks["RQ4_disagreement_precision_at_10pct"] = hitl.get("disagreement_at_10pct", {}).get("precision")
        checks["RQ4_random_precision_at_10pct"] = hitl.get("random_at_10pct", {}).get("precision")
        checks["RQ4_efficiency_ratio_10pct"] = hitl.get("efficiency_ratios", [{}, {}])[1].get("efficiency_disagreement_vs_random")

    # 4. RQ8: Retrieval
    retrieval = load_json("retrieval_summary.json")
    if retrieval:
        checks["RQ8_semantic_p10"] = retrieval.get("mean_semantic_precision")
        checks["RQ8_keyword_p10"] = retrieval.get("mean_keyword_precision")

    # 5. Complexity SCI Correlation
    sci = load_json("scene_complexity_summary.json")
    if sci:
        checks["RQ7_sci_spearman_r"] = sci.get("spearman_sci_vs_delta_fusion_mono")

    # 6. E2 & E4 Table Data (Synthetic Human Baseline)
    synthetic = load_json("synthetic_human_summary.json")
    if synthetic:
        checks["E2_macro_f1"] = synthetic.get("macro_f1")
        checks["E2_macro_agreement"] = synthetic.get("macro_agreement")

    # 7. Statistical Report
    stats = load_json("statistical_report_summary.json")
    if stats:
        checks["stats_wilcoxon_p"] = stats.get("key_comparison_monolithic_vs_fusion", {}).get("wilcoxon_p")
        checks["stats_cohens_d"] = stats.get("key_comparison_monolithic_vs_fusion", {}).get("cohens_d")
        checks["stats_fusion_mean"] = stats.get("key_comparison_monolithic_vs_fusion", {}).get("fusion_mean")
        checks["stats_mono_mean"] = stats.get("key_comparison_monolithic_vs_fusion", {}).get("monolithic_mean")

    # Output the report
    with open(OUTPUT_FILE, "w") as f:
        json.dump(checks, f, indent=4)
    
    print(f"✅ Fact-check report generated: {OUTPUT_FILE}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    fact_check()
