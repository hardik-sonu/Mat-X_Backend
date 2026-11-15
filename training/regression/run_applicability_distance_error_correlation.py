from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT = Path("training_results/applicability_prediction_error_validation_v1")
INPUT = ROOT / "matched_applicability_predictions.csv"
OUT = ROOT / "continuous_distance_error_analysis_v1"
OUT.mkdir(parents=True, exist_ok=True)
if not INPUT.exists():
    raise FileNotFoundError(INPUT)
df = pd.read_csv(INPUT)
required = [
    "target", "held_out_source", "variant",
    "grade_id", "heat_id",
    "nearest_training_distance", "absolute_error",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")
record_keys = [
    "target", "held_out_source", "variant", "heat_id"
]
if df.duplicated(record_keys).any():
    raise RuntimeError(
        "Duplicate heat IDs found within an assessment."
    )
numeric = ["nearest_training_distance", "absolute_error"]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if not np.isfinite(df[numeric].to_numpy()).all():
    raise RuntimeError("Non-finite distance or absolute-error values found.")
fold_keys = ["target", "held_out_source", "variant"]
grade_keys = fold_keys + ["grade_id"]
# One observation per grade per fold.
grade = (
    df.groupby(grade_keys, dropna=False)
      .agg(
          heat_count=("heat_id", "nunique"),
          mean_nearest_training_distance=(
              "nearest_training_distance", "mean"
          ),
          median_nearest_training_distance=(
              "nearest_training_distance", "median"
          ),
          mean_absolute_error=("absolute_error", "mean"),
          median_absolute_error=("absolute_error", "median"),
      )
      .reset_index()
)
if grade.duplicated(grade_keys).any():
    raise RuntimeError("Grade-level aggregation is not unique.")
grade_path = OUT / "grade_level_distance_error.csv"
grade.to_csv(grade_path, index=False)
# Grade-level Spearman correlation within each independent fold.
REPS = 10000
SEED = 20261114
rng = np.random.default_rng(SEED)
results = []
for key, group in grade.groupby(fold_keys, sort=True):
    x = group["mean_nearest_training_distance"].to_numpy(float)
    y = group["mean_absolute_error"].to_numpy(float)
    n = len(group)
    row = dict(zip(fold_keys, key))
    row["grade_count"] = int(n)
    row["bootstrap_repetitions"] = REPS
    if n < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        row.update({
            "spearman_rho": np.nan,
            "p_value": np.nan,
            "bootstrap_ci95_lower": np.nan,
            "bootstrap_ci95_upper": np.nan,
            "status": "INSUFFICIENT_GRADE_SUPPORT_OR_VARIATION",
        })
        results.append(row)
        continue
    observed = spearmanr(x, y)
    boot = np.empty(REPS, dtype=float)
    valid = 0
    attempts = 0
    max_attempts = REPS * 3
    while valid < REPS and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, n, size=n)
        xb = x[indices]
        yb = y[indices]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        value = spearmanr(xb, yb).statistic
        if np.isfinite(value):
            boot[valid] = value
            valid += 1
    if valid < REPS:
        raise RuntimeError(
            f"Could not complete bootstrap for fold {key}: "
            f"{valid}/{REPS}"
        )
    lower, upper = np.quantile(boot, [0.025, 0.975])
    row.update({
        "spearman_rho": float(observed.statistic),
        "p_value": float(observed.pvalue),
        "bootstrap_ci95_lower": float(lower),
        "bootstrap_ci95_upper": float(upper),
        "status": "COMPLETED",
    })
    results.append(row)
summary = pd.DataFrame(results)
# Holm family-wise adjustment across completed fold-level tests.
valid_mask = summary["p_value"].notna()
p = summary.loc[valid_mask, "p_value"].to_numpy(float)
adjusted = np.full(len(p), np.nan)
if len(p):
    order = np.argsort(p)
    sorted_p = p[order]
    m = len(sorted_p)
    sorted_adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)
    adjusted[order] = sorted_adjusted
summary["holm_adjusted_p_value"] = np.nan
summary.loc[valid_mask, "holm_adjusted_p_value"] = adjusted
summary["adjusted_significant"] = (
    summary["holm_adjusted_p_value"] < 0.05
)
summary_path = OUT / "fold_distance_error_spearman.csv"
summary.to_csv(summary_path, index=False)
report = {
    "input_rows": int(len(df)),
    "unique_heat_ids": int(df["heat_id"].nunique()),
    "grade_level_rows": int(len(grade)),
    "fold_count": int(len(summary)),
    "completed_folds": int(
        (summary["status"] == "COMPLETED").sum()
    ),
    "insufficient_folds": int(
        (summary["status"] != "COMPLETED").sum()
    ),
    "bootstrap_repetitions_per_completed_fold": REPS,
    "seed": SEED,
    "multiple_testing_method": "Holm",
    "family_alpha": 0.05,
    "significant_folds_after_adjustment": int(
        summary["adjusted_significant"].sum()
    ),
    "interpretation_limitations": [
        "Associations are descriptive and not causal.",
        "One grade contributes one observation per fold.",
        "Mean grade distance and mean grade error are used.",
        "Bootstrap intervals resample grades within each fold.",
        "Spearman p-values are asymptotic and Holm-adjusted.",
        "Results do not establish external validity.",
    ],
    "outputs": [
        "grade_level_distance_error.csv",
        "fold_distance_error_spearman.csv",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
print("=== GRADE-LEVEL AGGREGATION ===")
print("Rows:", len(grade))
print("Unique fold groups:", grade.groupby(fold_keys).ngroups)
print(grade.head(10).to_string(index=False))
print("\n=== DISTANCE–ERROR CORRELATIONS ===")
print(summary.to_string(index=False))
print("\n=== SUMMARY ===")
print(json.dumps(report, indent=2))
print("\nOutput directory:", OUT.resolve())
print("\n=== ANALYSIS COMPLETE ===")

