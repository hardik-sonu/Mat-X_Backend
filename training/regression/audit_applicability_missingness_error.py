import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
BASE = ROOT / "training_results" / "applicability_promoted_reference_audit" / "held_out_test_validation_v1"
for target in ["tensile_strength", "yield_strength"]:
    print("\n" + "=" * 75)
    print(target.upper())
    print("=" * 75)
    path = BASE / f"{target}_held_out_record_audit.csv"
    df = pd.read_csv(path)
    print("\nAVAILABLE COLUMNS:")
    print(df.columns.tolist())
    distance_col = next(
        (c for c in df.columns if "distance" in c.lower()),
        None
    )
    error_col = next(
        (c for c in df.columns if "absolute_error" in c.lower()),
        None
    )
    if distance_col is None or error_col is None:
        raise RuntimeError("Distance or absolute-error column not found.")
    groups = {
        "ALL_RECORDS": df,
        "NO_MISSING_FEATURES": df[df["missing_feature_count"] == 0],
        "WITH_MISSING_FEATURES": df[df["missing_feature_count"] > 0],
        "NO_UNSEEN_CATEGORIES": df[df["unseen_category_count"] == 0],
        "WITH_UNSEEN_CATEGORIES": df[df["unseen_category_count"] > 0],
    }
    print("\nDISTANCE / ERROR ASSOCIATION BY EVIDENCE GROUP")
    for name, subset in groups.items():
        valid = subset[[distance_col, error_col]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) >= 3:
            result = spearmanr(
                valid[distance_col],
                valid[error_col]
            )
            print(
                f"{name}: n={len(valid)}, "
                f"rho={result.statistic:.4f}, "
                f"p={result.pvalue:.6g}"
            )
        else:
            print(f"{name}: insufficient records ({len(valid)})")
    print("\nERROR BY DISTANCE THRESHOLD")
    for threshold in ["q90", "q95"]:
        flag = f"distance_{threshold}_exceeded"
        if flag not in df.columns:
            print(f"{flag}: not found")
            continue
        for exceeded in [False, True]:
            subset = df[df[flag] == exceeded]
            print(
                f"{threshold} exceeded={exceeded}: "
                f"n={len(subset)}, "
                f"mean_abs_error={subset[error_col].mean():.4f}, "
                f"median_abs_error={subset[error_col].median():.4f}"
            )
    print("\nMISSING-FEATURE COUNT VS ERROR")
    print(
        df.groupby("missing_feature_count")[error_col]
        .agg(["count", "mean", "median"])
        .to_string()
    )
print("\nREAD-ONLY EVIDENCE AUDIT COMPLETE.")
