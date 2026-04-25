import pandas as pd

df = pd.read_csv("/data/brhanu/thesis_project/results/agreement_scores_v8/agreement_v8_final_results.csv")
cols = [c for c in df.columns if "_" in c and c != "image_id" and c != "final_score"]
print("Pairwise Mean Agreement Scores:")
print(df[cols].mean().sort_values(ascending=False))
print(f"\nGlobal Mean Agreement: {df['final_score'].mean():.4f}")
