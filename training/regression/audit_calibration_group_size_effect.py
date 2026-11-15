from pathlib import Path
import pandas as pd
ROOT = Path(r"training_results\uncertainty_calibration_v1\calibration_residual_diagnostics")
for target in ["tensile_strength", "yield_strength"]:
    path = ROOT / f"{target}_individual_calibration_residuals.csv"
    df = pd.read_csv(path)
    grades = (
        df.groupby(["grade_id", "source"], dropna=False)
        .agg(
            calibration_records=("absolute_residual", "size"),
            maximum_absolute_residual=("absolute_residual", "max"),
            mean_absolute_residual=("absolute_residual", "mean"),
            median_absolute_residual=("absolute_residual", "median"),
        )
        .reset_index()
    )
    print("\n" + "=" * 90)
    print(target.upper(), "— CALIBRATION GROUP SIZE AUDIT")
    print("=" * 90)
    print("\nGROUP SIZE DISTRIBUTION:")
    print(grades["calibration_records"].value_counts().sort_index().to_string())
    print("\nGROUP SIZE VS MAXIMUM RESIDUAL:")
    print(
        grades.groupby("calibration_records")[
            "maximum_absolute_residual"
        ].agg(["count", "median", "mean", "max"]).to_string()
    )
    print("\nSPEARMAN CORRELATION:")
    print(
        grades[["calibration_records", "maximum_absolute_residual"]]
        .corr(method="spearman")
        .to_string()
    )
    print("\nLARGEST CALIBRATION GROUPS:")
    print(
        grades.nlargest(10, "calibration_records")[
            ["grade_id", "source", "calibration_records",
             "maximum_absolute_residual", "mean_absolute_residual"]
        ].to_string(index=False)
    )
    print("\nNIMS GROUP:")
    print(
        grades[grades["grade_id"] == "NIMS_low_alloy_other"].to_string(index=False)
    )
print("\nAUDIT COMPLETE — NO SOURCE DATA OR MODELS MODIFIED.")
