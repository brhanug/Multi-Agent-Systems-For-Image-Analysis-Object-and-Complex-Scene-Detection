#!/usr/bin/env python3
"""
run_gold_evaluation.py
-----------------------
E2 + E4: Measured ablation and scene complexity stratification against
gold (or synthetic) labels from labeling_worksheet.csv.

Produces:
  results/multi_agent/gold_evaluation_summary.json
  results/multi_agent/gold_evaluation_per_class.csv
  results/multi_agent/gold_evaluation_complexity_stratified.csv
  results/multi_agent/gold_evaluation_uncertainty_correlation.csv

Key metrics computed:
  - Per-agent precision/recall/F1 vs gold (leave-one-out ablation, E2)
  - Scene complexity stratification (5 bins) vs actual errors (E4)
  - Pearson correlation: uncertainty_score <-> human annotation error rate
  - Fusion vs monolithic vs individual-agent F1 @ gold labels
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
WORKSHEET     = BASE / "human_baseline_gold_kit" / "labeling_worksheet.csv"
AGENT_SCORES  = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
MA_SCORES     = BASE / "results" / "multi_agent" / "multi_agent_validation_scores.csv"
COMPLEXITY    = BASE / "results" / "multi_agent" / "scene_complexity_index.csv"
OUTPUT_DIR    = BASE / "results" / "multi_agent"

# 5 binary expert class columns
CLASS_COLS = ["label_teaching", "label_family", "label_playing", "label_landscape", "label_drawing"]

# Validation agent names → agent_comparison_scores.csv column names
VALIDATION_AGENTS = {
    "object":    "existing_pipeline_agent",
    "agreement": "agreement_agent",
    "scene":     "scene_agent",
    "vlm":       "vlm_agent",
}
FUSION_COL      = "comparison_fusion_score"
MONOLITHIC_COL  = "monolithic_pipeline_agent"

# ---------------------------------------------------------------------------


def normalize_id(name: str) -> str:
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1]
    p = p.rsplit(".", 1)[0]
    parts = p.split("_")
    if len(parts) >= 2 and parts[0].startswith("PPN"):
        return "_".join(parts[-2:])
    return p


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def safe_f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


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


# ---------------------------------------------------------------------------
def main() -> None:
    # ---------- 1. Load actual human labels --------------------------------
    HUMAN_LABELS_PATH = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
    if not HUMAN_LABELS_PATH.exists():
        print(f"❌ Human gold labels not found: {HUMAN_LABELS_PATH}", file=sys.stderr)
        sys.exit(1)

    ws = pd.read_csv(HUMAN_LABELS_PATH)
    ws["cvat_id"] = ws.index
    # Keep only the 801 reviewed images
    ws_valid = ws[ws["cvat_id"] <= 800].copy()
    ws_valid["_key"] = ws_valid["image_id"].astype(str).apply(normalize_id)
    is_synthetic = False
    print(f"📊 Worksheet rows (human reviewed): {len(ws_valid)}  |  Synthetic: {is_synthetic}")

    # Per-image gold label: 1 if expert annotated any scene, 0 if left blank
    ws_valid["gold_has_objects"] = (ws_valid["n_scene_labels"] > 0).astype(int)
    gold = ws_valid.set_index("_key")

    # ---------- 2. Load agent scores ---------------------------------------
    if not AGENT_SCORES.exists():
        print(f"❌ Agent scores not found: {AGENT_SCORES}", file=sys.stderr)
        sys.exit(1)

    agents = pd.read_csv(AGENT_SCORES)
    agents["_key"] = agents["image_id"].astype(str).apply(normalize_id)
    # De-duplicate (take first occurrence per image for this evaluation)
    agents = agents.drop_duplicates("_key").set_index("_key")

    # ---------- 3. Join gold + agents --------------------------------------
    joined = gold.join(agents, how="inner", lsuffix="_gold", rsuffix="_agent")
    n_matched = len(joined)
    print(f"🔗 Matched worksheet ↔ agent scores: {n_matched} images")
    if n_matched < 10:
        print("⚠️  Too few matches to produce meaningful metrics. Check image ID formats.")

    # ---------- 4. E2 — Leave-one-out ablation (per validation agent) ------
    gold_label = joined["gold_has_objects"].astype(float)

    agent_metrics: dict[str, dict] = {}
    for name, col in VALIDATION_AGENTS.items():
        if col not in joined.columns:
            print(f"   skip {name} (column '{col}' missing)")
            continue
        m = binary_prf(joined[col].astype(float), gold_label)
        m["agent"] = name
        m["column"] = col
        agent_metrics[name] = m

    # Fusion and monolithic
    for name, col in [("fusion", FUSION_COL), ("monolithic", MONOLITHIC_COL)]:
        if col in joined.columns:
            m = binary_prf(joined[col].astype(float), gold_label)
            m["agent"] = name
            m["column"] = col
            agent_metrics[name] = m

    print("\n📊 E2 — Agent evaluation vs gold labels:")
    header = f"{'Agent':15s} | {'Precision':9s} | {'Recall':6s} | {'F1':6s} | {'Accuracy':8s} | n_matched"
    print(header)
    print("-" * len(header))
    for nm, m in agent_metrics.items():
        print(f"{nm:15s} | {m['precision']:.4f}    | {m['recall']:.4f} | {m['f1']:.4f} | {m['accuracy']:.4f}   | {n_matched}")

    # ---------- 5. Per-class breakdown ------------------------------------
    per_class_rows: list[dict] = []
    for cls in CLASS_COLS:
        if cls not in gold.columns:
            continue
        gold_cls = joined[cls].astype(float)
        for name, col in VALIDATION_AGENTS.items():
            if col not in joined.columns:
                continue
            m = binary_prf(joined[col].astype(float), gold_cls)
            per_class_rows.append({"class": cls, "agent": name, **m})
        for name, col in [("fusion", FUSION_COL), ("monolithic", MONOLITHIC_COL)]:
            if col in joined.columns:
                m = binary_prf(joined[col].astype(float), gold_cls)
                per_class_rows.append({"class": cls, "agent": name, **m})

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_out = OUTPUT_DIR / "gold_evaluation_per_class.csv"
    per_class_df.to_csv(per_class_out, index=False)
    print(f"\n✅ Per-class metrics: {per_class_out}")

    # ---------- 6. E4 — Complexity stratification -------------------------
    complexity_rows: list[dict] = []
    if COMPLEXITY.exists():
        comp = pd.read_csv(COMPLEXITY, low_memory=False)
        comp_key = "image_id" if "image_id" in comp.columns else comp.columns[0]
        comp["_key"] = comp[comp_key].astype(str).apply(normalize_id)
        comp = comp.drop_duplicates("_key").set_index("_key")

        complexity_col = next(
            (c for c in ["scene_complexity_index", "complexity_index", "sci_index"] if c in comp.columns), None)
        if complexity_col is None:
            print(f"⚠️  No complexity column found — skipping E4")
        else:
            joined_c = joined.join(comp[[complexity_col]], how="left")
            joined_c = joined_c.rename(columns={complexity_col: "complexity_index"})
            temp_bins = pd.qcut(
                joined_c["complexity_index"].fillna(joined_c["complexity_index"].median()),
                q=5, duplicates="drop"
            )
            n_bins = len(temp_bins.cat.categories)
            labels = ["very_low", "low", "medium", "high", "very_high"][:n_bins]
            joined_c["complexity_bin"] = pd.qcut(
                joined_c["complexity_index"].fillna(joined_c["complexity_index"].median()),
                q=5, labels=labels, duplicates="drop"
            )
            for bin_label, grp in joined_c.groupby("complexity_bin", observed=True):
                gold_grp = grp["gold_has_objects"].astype(float)
                row: dict = {"complexity_bin": str(bin_label), "n": len(grp)}
                for name, col in {**VALIDATION_AGENTS, "fusion": FUSION_COL, "monolithic": MONOLITHIC_COL}.items():
                    if col in grp.columns:
                        m = binary_prf(grp[col].astype(float), gold_grp)
                        row[f"{name}_f1"] = m["f1"]
                complexity_rows.append(row)
            comp_df = pd.DataFrame(complexity_rows)
            comp_out = OUTPUT_DIR / "gold_evaluation_complexity_stratified.csv"
            comp_df.to_csv(comp_out, index=False)
            print(f"✅ Complexity-stratified metrics: {comp_out}")
    else:
        print("⚠️  complexity_index.csv not found — skipping E4 complexity stratification")

    # ---------- 7. Uncertainty ↔ error correlation -----------------------
    if MA_SCORES.exists():
        ma = pd.read_csv(MA_SCORES, low_memory=False)
        ma["_key"] = ma["image_id"].astype(str).apply(normalize_id)
        ma = ma.drop_duplicates("_key").set_index("_key")

        if "uncertainty_score" in ma.columns:
            joined_u = joined.join(ma[["uncertainty_score"]], how="left")
            joined_u = joined_u.dropna(subset=["uncertainty_score"])

            # Error = system prediction disagrees with gold
            if FUSION_COL in joined_u.columns:
                joined_u["system_pred"] = (joined_u[FUSION_COL] >= 0.5).astype(int)
            else:
                joined_u["system_pred"] = (joined_u[MONOLITHIC_COL] >= 0.5).astype(int)
            joined_u["is_error"] = (joined_u["system_pred"] != joined_u["gold_has_objects"]).astype(int)

            r = pearson(
                joined_u["uncertainty_score"].tolist(),
                joined_u["is_error"].tolist(),
            )
            print(f"\n📈 Uncertainty ↔ error Pearson r = {r:.4f}  (n={len(joined_u)})")
            print("   Interpretation: higher r → uncertainty is a good predictor of errors")

            unc_corr_df = joined_u[["uncertainty_score", "is_error", "gold_has_objects"]].reset_index()
            unc_corr_out = OUTPUT_DIR / "gold_evaluation_uncertainty_correlation.csv"
            unc_corr_df.to_csv(unc_corr_out, index=False)
            print(f"✅ Uncertainty-correlation data: {unc_corr_out}")
        else:
            print("⚠️  'uncertainty_score' column missing in multi_agent_validation_scores.csv")
    else:
        print("⚠️  multi_agent_validation_scores.csv not found — skipping uncertainty correlation")

    # ---------- 8. Summary JSON ------------------------------------------
    summary: dict = {
        "n_gold_images":     len(ws_valid),
        "n_matched":         n_matched,
        "is_synthetic":      bool(is_synthetic),
        "agent_metrics":     {k: {kk: (round(v, 4) if isinstance(v, float) else v)
                                  for kk, v in m.items()}
                              for k, m in agent_metrics.items()},
        "complexity_bins_computed": len(complexity_rows),
        "best_agent_by_f1": max(agent_metrics, key=lambda k: agent_metrics[k]["f1"])
                            if agent_metrics else "n/a",
    }
    out_json = OUTPUT_DIR / "gold_evaluation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary: {out_json}")

    best = summary["best_agent_by_f1"]
    print(f"\n🏆 Best agent by F1 vs gold: {best} "
          f"(F1 = {agent_metrics[best]['f1']:.4f})" if best != "n/a" else "")


if __name__ == "__main__":
    main()
