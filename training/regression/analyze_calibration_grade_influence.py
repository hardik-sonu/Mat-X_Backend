from pathlib import Path
import math
import numpy as np
import pandas as pd
ROOT = Path(
    r"C:\Users\Ahmadcomputer\MAT-X\training_results"
    r"\uncertainty_calibration_v1\calibration_residual_diagnostics"
)
def conformal_q(values, confidence):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    rank = math.ceil((n + 1) * confidence)
    if n == 0 or rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])
for target in ["tensile_strength", "yield_strength"]:
    path = ROOT / f"{target}_calibration_grade_diagnostics.csv"
    df = pd.read_csv(path)
    scores = df["maximum_absolute_residual"].to_numpy(dtype=float)
    counts = df["calibration_record_count"].to_numpy(dtype=int)
    print("\n" + "=" * 85)
    print(target.upper(), "CALIBRATION INFLUENCE ANALYSIS")
    print("=" * 85)
    for confidence in [0.90, 0.95]:
        baseline = conformal_q(scores, confidence)
        leave_one_out = []
        for i, row in df.iterrows():
            reduced = np.delete(scores, i)
            q = conformal_q(reduced, confidence)
            leave_one_out.append({
                "removed_grade": row["grade_id"],
                "removed_record_count": int(row["calibration_record_count"]),
                "removed_score": float(row["maximum_absolute_residual"]),
                "recomputed_quantile": q,
                "quantile_change": q - baseline,
            })
        loo = pd.DataFrame(leave_one_out)
        loo_path = ROOT / (
            f"{target}_leave_one_grade_out_q{int(confidence*100)}.csv"
        )
        loo.to_csv(loo_path, index=False)
        print(f"\nQ{int(confidence*100)} baseline: {baseline:.6f}")
        print("LOO minimum:", loo["recomputed_quantile"].min())
        print("LOO maximum:", loo["recomputed_quantile"].max())
        print("Largest absolute quantile changes:")
        print(
            loo.assign(
                absolute_change=loo["quantile_change"].abs()
            )
            .sort_values("absolute_change", ascending=False)
            .head(10)[
                [
                    "removed_grade",
                    "removed_record_count",
                    "removed_score",
                    "recomputed_quantile",
                    "quantile_change",
                ]
            ]
            .to_string(index=False)
        )
    print("\nRECORD-COUNT STRATA:")
    df["record_count_stratum"] = pd.cut(
        df["calibration_record_count"],
        bins=[0, 1, 3, 10, np.inf],
        labels=["1 record", "2-3 records", "4-10 records", "over 10"],
    )
    print(
        df.groupby("record_count_stratum", observed=False)
        .agg(
            grades=("grade_id", "count"),
            median_score=("maximum_absolute_residual", "median"),
            mean_score=("maximum_absolute_residual", "mean"),
            maximum_score=("maximum_absolute_residual", "max"),
        )
        .to_string()
    )
print("\nLEAVE-ONE-GRADE-OUT ANALYSIS COMPLETE.")
print("Outputs:", ROOT)
