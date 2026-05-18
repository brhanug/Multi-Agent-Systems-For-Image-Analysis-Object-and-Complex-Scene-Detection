#!/usr/bin/env python3
import pandas as pd
import numpy as np

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1]
    p = p.rsplit(".", 1)[0]
    parts = p.split("_")
    if len(parts) >= 2 and parts[0].startswith("PPN"):
        return "_".join(parts[-2:])
    return p

def binary_prf(pred: pd.Series, true: pd.Series):
    tp = int(((pred >= 0.5) & (true == 1)).sum())
    fp = int(((pred >= 0.5) & (true == 0)).sum())
    fn = int(((pred < 0.5) & (true == 1)).sum())
    tn = int(((pred < 0.5) & (true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(pred) if len(pred) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

def main():
    print("📊 Stratifying real expert validation by scene complexity...")
    
    # Load expert labels
    expert = pd.read_csv("human_baseline_gold_kit/gold_labels_human.csv")
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    # Load agent scores
    agents = pd.read_csv("results/multi_agent/agent_comparison_scores.csv")
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    # Join
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    
    # Load complexity
    comp = pd.read_csv("results/multi_agent/scene_complexity_index.csv")
    comp["image_id"] = comp["image_id"].apply(normalize_id)
    comp = comp.drop_duplicates("image_id").set_index("image_id")
    
    joined_c = joined.join(comp[["scene_complexity_index"]], on="image_id", how="left")
    joined_c = joined_c.rename(columns={"scene_complexity_index": "complexity_index"})
    
    # Assign complexity bins
    filled_idx = joined_c["complexity_index"].fillna(joined_c["complexity_index"].median())
    # Compute cuts dynamically to avoid ValueError with duplicates="drop"
    cuts, bins = pd.qcut(filled_idx, q=5, retbins=True, duplicates="drop")
    n_bins = len(bins) - 1
    labels = ["very_low", "low", "medium", "high", "very_high"][:n_bins]
    joined_c["complexity_bin"] = pd.qcut(filled_idx, q=5, labels=labels, duplicates="drop")
    
    VALIDATION_AGENTS = {
        "object": "existing_pipeline_agent",
        "agreement": "agreement_agent",
        "scene": "scene_agent",
        "vlm": "vlm_agent",
        "fusion": "comparison_fusion_score",
        "monolithic": "monolithic_pipeline_agent"
    }
    
    print("\n📈 REAL COMPLEXITY-STRATIFIED F1:")
    header = f"{'Complexity Bin':15s} | {'n':4s} | {'Object F1':9s} | {'Agreement F1':12s} | {'Scene F1':8s} | {'VLM F1':6s} | {'Fusion F1':9s} | {'Mono F1':7s}"
    print(header)
    print("-" * len(header))
    
    for bin_label, grp in joined_c.groupby("complexity_bin", observed=True):
        gold_grp = grp["gold_has_scene"].astype(float)
        row = {"complexity_bin": str(bin_label), "n": len(grp)}
        for name, col in VALIDATION_AGENTS.items():
            m = binary_prf(grp[col].astype(float), gold_grp)
            row[name] = m["f1"]
        print(f"{str(bin_label):15s} | {len(grp):4d} | {row['object']:.4f}    | {row['agreement']:.4f}       | {row['scene']:.4f}   | {row['vlm']:.4f} | {row['fusion']:.4f}    | {row['monolithic']:.4f}")

if __name__ == "__main__":
    main()
