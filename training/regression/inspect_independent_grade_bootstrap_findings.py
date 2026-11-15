from pathlib import Path
import pandas as pd
import json
base = (
    Path.cwd()
    / "training_results"
    / "applicability_prediction_error_validation_v1"
    / "independent_grade_bootstrap_v1"
)
path = base / "independent_grade_bootstrap_contrasts.csv"
df = pd.read_csv(path)
print("=== ALL DISTINGUISHABLE CONTRASTS ===")
selected = df[
    df["interpretation"].isin([
        "HIGHER_ERROR_THAN_IN_DOMAIN",
        "LOWER_ERROR_THAN_IN_DOMAIN",
    ])
].copy()
columns = [
    "target",
    "held_out_source",
    "variant",
    "comparison_status",
    "in_domain_grades",
    "comparison_grades",
    "mean_mae_difference",
    "ci95_lower",
    "ci95_upper",
    "interpretation",
]
print(
    selected[columns]
    .sort_values([
        "interpretation",
        "target",
        "held_out_source",
        "variant",
    ])
    .to_string(index=False)
)
print("\n=== ELIGIBLE COMPARISONS ===")
eligible = df[
    df["bootstrap_repetitions"] > 0
]
print("Eligible:", len(eligible))
print("Higher:", (
    eligible["interpretation"]
    == "HIGHER_ERROR_THAN_IN_DOMAIN"
).sum())
print("Lower:", (
    eligible["interpretation"]
    == "LOWER_ERROR_THAN_IN_DOMAIN"
).sum())
print("No clear difference:", (
    eligible["interpretation"]
    == "NO_CLEAR_DIFFERENCE"
).sum())
print("\n=== CONFIDENCE INTERVAL INTEGRITY ===")
valid = (
    eligible["ci95_lower"].notna()
    & eligible["ci95_upper"].notna()
    & (eligible["ci95_lower"] <= eligible["ci95_upper"])
)
print("Eligible rows:", len(eligible))
print("Valid intervals:", int(valid.sum()))
print("Invalid intervals:", int((~valid).sum()))
if len(eligible) != 19 or not valid.all():
    raise RuntimeError("Bootstrap result integrity check failed.")
print("\n=== SUMMARY JSON ===")
summary = json.loads(
    (base / "summary.json").read_text(encoding="utf-8")
)
print(json.dumps(summary, indent=2))
print("\n=== INSPECTION COMPLETE ===")
