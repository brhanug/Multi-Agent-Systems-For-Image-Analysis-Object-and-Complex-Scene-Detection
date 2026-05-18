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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

def main():
    print("📊 Computing real agent validation metrics directly against CVAT expert labels...")
    
    # Load expert labels
    expert = pd.read_csv("human_baseline_gold_kit/gold_labels_human.csv")
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    
    # Build binary gold scene presence label: 1 if expert annotated any scene, 0 if left blank
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    # Load agent scores
    agents = pd.read_csv("results/multi_agent/agent_comparison_scores.csv")
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    # Join
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"✅ Matched exactly {len(joined)} reviewed images.")
    
    # Run evaluation
    gold_label = joined["gold_has_scene"]
    
    VALIDATION_AGENTS = {
        "Object Agent":    "existing_pipeline_agent",
        "Agreement Agent": "agreement_agent",
        "Scene Agent":     "scene_agent",
        "VLM Agent":       "vlm_agent",
        "Monolithic Baseline": "monolithic_pipeline_agent",
        "Coordinator Fusion": "comparison_fusion_score"
    }
    
    results = []
    print("\n📈 REAL METRICS AGAINST EXPERT BASILINE:")
    header = f"{'Agent / Strategy':23s} | {'Precision':9s} | {'Recall':6s} | {'F1':6s} | {'Accuracy':8s}"
    print(header)
    print("-" * len(header))
    
    for name, col in VALIDATION_AGENTS.items():
        m = binary_prf(joined[col].astype(float), gold_label)
        results.append({
            "Agent": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Accuracy": m["accuracy"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"]
        })
        print(f"{name:23s} | {m['precision']:.4f}    | {m['recall']:.4f} | {m['f1']:.4f} | {m['accuracy']:.4f}")

if __name__ == "__main__":
    main()
