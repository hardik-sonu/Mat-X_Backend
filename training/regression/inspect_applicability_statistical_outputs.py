from pathlib import Path
import json
import pandas as pd
root = Path(
    "training_results/applicability_prediction_error_validation_v1"
)
files = {
    "matched_predictions": root / "matched_applicability_predictions.csv",
    "grade_error_distributions": root / "statistical_robustness_v2" / "grade_error_distributions.csv",
    "adjusted_contrasts": root / "statistical_robustness_v2" / "multiplicity_adjusted_contrasts_100k.csv",
    "bootstrap_comparison": root / "statistical_robustness_v2" / "comparison_against_5k_bootstrap.csv",
    "summary": root / "statistical_robustness_v2" / "summary.json",
}
for name, path in files.items():
    print(f"\n=== {name.upper()} ===")
    if not path.exists():
        print("MISSING:", path)
        continue
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2))
        continue
    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    if name == "grade_error_distributions":
        print("\nPreview:")
        print(df.head(8).to_string(index=False))
    elif name == "adjusted_contrasts":
        print("\nComparison support and effect sizes:")
        cols = [
            c for c in [
                "target", "held_out_source", "variant",
                "comparison_status", "in_domain_grades",
                "comparison_grades", "mean_mae_difference",
                "bonferroni_ci95_lower",
                "bonferroni_ci95_upper", "interpretation"
            ] if c in df.columns
        ]
        print(df[cols].to_string(index=False))
print("\n=== OUTPUT INSPECTION COMPLETE ===")
