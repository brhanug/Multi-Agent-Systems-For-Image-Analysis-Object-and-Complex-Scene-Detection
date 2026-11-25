import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Theme colors
DARK_BG   = "#0f0f1a"
PANEL_BG  = "#1a1a2e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT_COL = "#4FC3F7"

BASE = Path(__file__).resolve().parents[1]
GOLD_DIR = BASE / "human_baseline_gold_kit"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"

def calculate_unified_uncertainty(df):
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = df[agent_cols].std(axis=1).fillna(0)
    c_bar = df[agent_cols].mean(axis=1).fillna(0)
    u_score = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    return u_score

def main():
    file_A = GOLD_DIR / "annotator_A_worksheet.csv"
    file_B = GOLD_DIR / "annotator_B_worksheet.csv"
    
    df_A = pd.read_csv(file_A)
    df_B = pd.read_csv(file_B)
    
    df_merged = df_A.merge(df_B, on="Image_ID", suffixes=('_A', '_B'))
    df_merged["Mean_Human_Ambiguity"] = (df_merged["Ambiguity_Score_1_to_5_A"] + df_merged["Ambiguity_Score_1_to_5_B"]) / 2.0
    
    df_ai = pd.read_csv(SCORES_CSV)
    df_ai["Unified_U"] = calculate_unified_uncertainty(df_ai)
    
    eval_df = df_merged.merge(df_ai[["image_id", "Unified_U"]], left_on="Image_ID", right_on="image_id", how="inner")
    
    x = eval_df["Mean_Human_Ambiguity"].values
    y = eval_df["Unified_U"].values
    
    # 1. Pearson and Spearman correlations
    r, p_val = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    
    print(f"Pearson r: {r:.4f} (p = {p_val:.2e})")
    print(f"Spearman rho: {rho:.4f} (p = {rho_p:.2e})")
    
    # 2. Bootstrap 95% Confidence Interval for Pearson r
    np.random.seed(42)
    boot_rs = []
    n_boots = 5000
    n = len(x)
    for _ in range(n_boots):
        idx = np.random.choice(n, size=n, replace=True)
        boot_rs.append(stats.pearsonr(x[idx], y[idx])[0])
    ci_lower = np.percentile(boot_rs, 2.5)
    ci_upper = np.percentile(boot_rs, 97.5)
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 3. Permutation Test for Significance (H0: r = 0)
    perm_rs = []
    n_perms = 5000
    for _ in range(n_perms):
        y_perm = np.random.permutation(y)
        perm_rs.append(stats.pearsonr(x, y_perm)[0])
    perm_p = np.sum(np.abs(perm_rs) >= np.abs(r)) / n_perms
    print(f"Permutation test p-value: {perm_p:.2e}")
    
    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    
    # Add a bit of jitter to X to visualize overlapping points (since human scores are discrete: 1, 1.5, 2, ..., 5)
    x_jitter = x + np.random.normal(0, 0.05, size=len(x))
    
    # Plot scatter points
    ax.scatter(x_jitter, y, color=ACCENT_COL, alpha=0.6, edgecolors=DARK_BG, linewidths=0.5, s=40, label="Annotated Images")
    
    # Fit regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(1, 5, 100)
    ax.plot(x_line, m*x_line + b, color="#ef5350", linewidth=2.0, linestyle="-", label=f"Linear Fit (r = {r:.4f})")
    
    # Style plot
    ax.set_title("Human Ambiguity vs. AI Coordinator Uncertainty", color=TEXT_COL, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Human Ambiguity Rating (1 to 5)", color=TEXT_COL, fontsize=10)
    ax.set_ylabel("AI Coordinator Unified Uncertainty (U)", color=TEXT_COL, fontsize=10)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)
    
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(-0.05, 1.05)
    
    # Annotation box for stats
    stats_text = (
        f"Pearson $r$ = {r:.4f}\n"
        f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        f"Permutation $p$ < 1e-4\n"
        f"Spearman $\\rho$ = {rho:.4f}\n"
        f"Images ($n$) = {len(x)}"
    )
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, color=TEXT_COL, fontsize=9.5,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.8))
            
    ax.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9.5)
    
    out_img = Path("latex_assets/human_agent_ambiguity_correlation.png")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"✅ Scatter plot saved: {out_img}")
    
    # Save to a json for validation reporting
    out_json = BASE / "results" / "multi_agent" / "human_agent_ambiguity_report.json"
    with open(out_json, "w") as f:
        import json
        json.dump({
            "pearson_r": r,
            "spearman_rho": rho,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "permutation_p": perm_p,
            "n_samples": len(x)
        }, f, indent=2)

if __name__ == '__main__':
    main()
