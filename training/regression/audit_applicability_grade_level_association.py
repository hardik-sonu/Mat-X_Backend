import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
BASE = ROOT / "training_results" / "applicability_promoted_reference_audit" / "held_out_test_validation_v1"
for target in ["tensile_strength", "yield_strength"]:
    print("\n" + "=" * 75)
    print(target.upper(), "- GRADE-LEVEL AUDIT")
    print("=" * 75)
    df = pd.read_csv(BASE / f"{target}_held_out_record_audit.csv")
    grade = (
        df.groupby("grade_id")
        .agg(
            records=("absolute_error", "size"),
            mean_distance=("nearest_training_distance", "mean"),
            maximum_distance=("nearest_training_distance", "max"),
            mean_absolute_error=("absolute_error", "mean"),
            maximum_absolute_error=("absolute_error", "max"),
            mean_missing_features=("missing_feature_count", "mean"),
            maximum_missing_features=("missing_feature_count", "max"),
            records_with_missing=("missing_feature_count", lambda x: (x > 0).sum()),
            records_with_unseen_categories=("unseen_category_count", lambda x: (x > 0).sum()),
            q90_exceeded=("distance_q90_exceeded", "any"),
            q95_exceeded=("distance_q95_exceeded", "any"),
        )
        .reset_index()
    )
    print("\nGRADE COUNTS:", len(grade))
    for distance_col in ["mean_distance", "maximum_distance"]:
        for error_col in ["mean_absolute_error", "maximum_absolute_error"]:
            valid = grade[[distance_col, error_col]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            result = spearmanr(valid[distance_col], valid[error_col])
            print(
                f"{distance_col} vs {error_col}: "
                f"n={len(valid)}, "
                f"rho={result.statistic:.4f}, "
                f"p={result.pvalue:.6g}"
            )
    print("\nGRADE-LEVEL THRESHOLD COMPARISON")
    for threshold in ["q90_exceeded", "q95_exceeded"]:
        for flag in [False, True]:
            subset = grade[grade[threshold] == flag]
            print(
                f"{threshold}={flag}: "
                f"grades={len(subset)}, "
                f"mean_grade_error="
                f"{subset['mean_absolute_error'].mean():.4f}, "
                f"median_grade_error="
                f"{subset['mean_absolute_error'].median():.4f}, "
                f"mean_max_error="
                f"{subset['maximum_absolute_error'].mean():.4f}"
            )
    print("\nTOP 15 GRADES BY MAXIMUM DISTANCE")
    print(
        grade.sort_values("maximum_distance", ascending=False)
        .head(15)
        .to_string(index=False)
    )
    output = BASE / f"{target}_grade_level_association_audit.csv"
    grade.to_csv(output, index=False)
    print("\nSaved:", output)
print("\nGRADE-LEVEL AUDIT COMPLETE. Source data and models were not modified.")
