import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
OUT = ROOT / "training_results" / "regression"
INPUT = OUT / "matx_elongation_feature_ablation_predictions.csv"
N_BOOTSTRAP = 2000
SEED = 42
rng = np.random.default_rng(SEED)
df = pd.read_csv(INPUT)
models = {
    "A": "A_baseline_absolute_error",
    "C": "C_plus_yield_absolute_error",
    "D": "D_plus_ratio_absolute_error",
}
required = ["source", "grade_id", "actual"] + list(models.values())
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
if df[required].isna().any().any():
    raise ValueError("Missing values detected.")
for col in models.values():
    if (df[col] < 0).any():
        raise ValueError(f"Negative absolute error in {col}")
def metrics(x):
    return {
        "MAE": float(np.mean(x)),
        "Median_AE": float(np.median(x)),
        "P90_AE": float(np.quantile(x, .90)),
        "P95_AE": float(np.quantile(x, .95)),
        "Max_AE": float(np.max(x)),
    }
def paired_bootstrap(data, model_a, model_b):
    # Cluster bootstrap by grade: retain all records for each sampled grade.
    grades = data["grade_id"].drop_duplicates().to_numpy()
    groups = {
        g: data.loc[data["grade_id"] == g, [model_a, model_b]].to_numpy()
        for g in grades
    }
    differences = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(grades, size=len(grades), replace=True)
        chunks = [groups[g] for g in sampled]
        sample = np.concatenate(chunks, axis=0)
        differences[b] = np.mean(sample[:, 0] - sample[:, 1])
    return {
        "mean_MAE_improvement": float(
            data[model_a].mean() - data[model_b].mean()
        ),
        "CI95_lower": float(np.quantile(differences, .025)),
        "CI95_upper": float(np.quantile(differences, .975)),
        "bootstrap_probability_b_better": float(
            np.mean(differences > 0)
        ),
        "bootstrap_replicates": N_BOOTSTRAP,
        "resampling_unit": "grade_id cluster",
    }
# Overall metrics
overall_rows = []
for model, col in models.items():
    row = {"model": model, "records": len(df)}
    row.update(metrics(df[col].to_numpy()))
    overall_rows.append(row)
overall = pd.DataFrame(overall_rows)
# Pooled paired cluster bootstrap
comparisons = [("A", "C"), ("A", "D"), ("C", "D")]
pooled_rows = []
for a, b in comparisons:
    result = paired_bootstrap(df, models[a], models[b])
    result["comparison"] = f"{b} vs {a}"
    pooled_rows.append(result)
pooled = pd.DataFrame(pooled_rows)
# Source-specific metrics and cluster bootstrap
source_metric_rows = []
source_bootstrap_rows = []
for source, part in df.groupby("source"):
    for model, col in models.items():
        row = {
            "source": source,
            "model": model,
            "records": len(part),
            "grades": part["grade_id"].nunique(),
        }
        row.update(metrics(part[col].to_numpy()))
        source_metric_rows.append(row)
    for a, b in comparisons:
        result = paired_bootstrap(part, models[a], models[b])
        result["source"] = source
        result["comparison"] = f"{b} vs {a}"
        source_bootstrap_rows.append(result)
source_metrics = pd.DataFrame(source_metric_rows)
source_bootstrap = pd.DataFrame(source_bootstrap_rows)
# Record-level wins
win_rows = []
for a, b in comparisons:
    diff = df[models[a]] - df[models[b]]
    win_rows.append({
        "comparison": f"{b} vs {a}",
        "records": len(diff),
        "b_wins": int((diff > 0).sum()),
        "a_wins": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "b_win_percent": float((diff > 0).mean() * 100),
    })
wins = pd.DataFrame(win_rows)
# Error tails
tail_rows = []
for model, col in models.items():
    x = df[col].to_numpy()
    for threshold in [5, 10, 15, 20]:
        tail_rows.append({
            "model": model,
            "threshold": threshold,
            "count_above": int((x > threshold).sum()),
            "percent_above": float((x > threshold).mean() * 100),
        })
tails = pd.DataFrame(tail_rows)
# Worst D cases, with all three model errors
worst_cols = [
    "source", "grade_id", "heat_id", "steel_family", "actual",
    *models.values()
]
worst = df[worst_cols].copy().sort_values(
    models["D"], ascending=False
).head(25)
# Display
print("=" * 90)
print("MAT-X ELONGATION PAIRED GRADE-CLUSTER BOOTSTRAP")
print("=" * 90)
print("\n1. OVERALL ERROR METRICS")
print(overall.to_string(index=False))
print("\n2. POOLED PAIRED BOOTSTRAP COMPARISONS")
print("Positive improvement means the second model has lower MAE.")
print(pooled.to_string(index=False))
print("\n3. SOURCE-WISE ERROR METRICS")
print(source_metrics.to_string(index=False))
print("\n4. SOURCE-WISE BOOTSTRAP COMPARISONS")
print(source_bootstrap.to_string(index=False))
print("\n5. RECORD-LEVEL MODEL WINS")
print(wins.to_string(index=False))
print("\n6. ERROR-TAIL ANALYSIS")
print(tails.to_string(index=False))
print("\n7. WORST 25 MODEL D PREDICTIONS")
print(worst.to_string(index=False))
# Save
outputs = {
    "matx_elongation_bootstrap_overall_metrics.csv": overall,
    "matx_elongation_bootstrap_comparisons.csv": pooled,
    "matx_elongation_bootstrap_source_metrics.csv": source_metrics,
    "matx_elongation_bootstrap_source_comparisons.csv": source_bootstrap,
    "matx_elongation_bootstrap_record_wins.csv": wins,
    "matx_elongation_bootstrap_error_tails.csv": tails,
    "matx_elongation_bootstrap_worst_cases.csv": worst,
}
for filename, table in outputs.items():
    table.to_csv(OUT / filename, index=False)
report = {
    "experiment": "MAT-X elongation paired grade-cluster bootstrap",
    "records": int(len(df)),
    "grades": int(df["grade_id"].nunique()),
    "sources": sorted(df["source"].unique().tolist()),
    "bootstrap_replicates": N_BOOTSTRAP,
    "confidence_level": 0.95,
    "random_seed": SEED,
    "resampling": "paired grade-cluster bootstrap within pooled and source subsets",
    "overall_metrics": overall.to_dict(orient="records"),
    "pooled_comparisons": pooled.to_dict(orient="records"),
    "source_metrics": source_metrics.to_dict(orient="records"),
    "source_comparisons": source_bootstrap.to_dict(orient="records"),
    "record_wins": wins.to_dict(orient="records"),
    "error_tails": tails.to_dict(orient="records"),
    "training_performed": False,
    "models_modified": False,
    "original_dataset_modified": False,
    "promotion_status": "NOT_PROMOTED",
}
with open(
    OUT / "matx_elongation_bootstrap_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n8. OUTPUT FILES")
for filename in outputs:
    print(OUT / filename)
print(OUT / "matx_elongation_bootstrap_report.json")
print("\nBOOTSTRAP ANALYSIS COMPLETED SUCCESSFULLY.")
