from pathlib import Path
import math
import numpy as np
import pandas as pd
ROOT = Path("training_results/uncertainty_calibration_v1")
AUDITS = [
    ("tensile_strength",
     ROOT / "tensile_strength_baseline_calibration_grade_scores.csv"),
    ("yield_strength",
     ROOT / "yield_strength_baseline_calibration_grade_scores.csv"),
]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
dataset = pd.read_csv(
    DATASET, low_memory=False, dtype={"grade_id": str}
)
dataset["grade_id"] = dataset["grade_id"].astype(str).str.strip()
def conformal_quantile(values, confidence):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * confidence)
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])
for target, path in AUDITS:
    print("\n" + "=" * 100)
    print("TARGET:", target)
    print("=" * 100)
    scores = pd.read_csv(path, dtype={"grade_id": str})
    scores["grade_id"] = scores["grade_id"].astype(str).str.strip()
    scores["maximum_absolute_residual"] = pd.to_numeric(
        scores["maximum_absolute_residual"], errors="coerce"
    )
    source_map = (
        dataset.groupby("grade_id")["source"]
        .agg(lambda x: " ; ".join(sorted(x.dropna().astype(str).unique())))
        .rename("source")
        .reset_index()
    )
    audit = scores.merge(source_map, on="grade_id", how="left")
    print("\nSOURCE COUNTS:")
    print(audit["source"].fillna("UNMATCHED").value_counts().to_string())
    print("\nSOURCE-SPECIFIC CALIBRATION:")
    for source, group in audit.groupby("source", dropna=False):
        values = group["maximum_absolute_residual"].dropna().to_numpy()
        print(f"\nSOURCE: {source}")
        print("Grades:", len(values))
        print("Median score:", float(np.median(values)) if len(values) else None)
        for confidence in (0.90, 0.95):
            q = conformal_quantile(values, confidence)
            print(f"{confidence:.0%} quantile:", q)
        print("Top five grades:")
        print(
            group.sort_values(
                "maximum_absolute_residual", ascending=False
            )[["grade_id", "maximum_absolute_residual"]]
            .head(5).to_string(index=False)
        )
    print("\nOVERALL SOURCE-EXCLUSION SENSITIVITY:")
    all_values = audit["maximum_absolute_residual"].dropna().to_numpy()
    for confidence in (0.90, 0.95):
        print(
            f"Original overall {confidence:.0%}:",
            conformal_quantile(all_values, confidence)
        )
    for source in sorted(audit["source"].dropna().unique()):
        remaining = audit.loc[
            audit["source"] != source,
            "maximum_absolute_residual"
        ].dropna().to_numpy()
        print(f"\nExcluding {source}: remaining grades={len(remaining)}")
        for confidence in (0.90, 0.95):
            print(
                f"  {confidence:.0%}:",
                conformal_quantile(remaining, confidence)
            )
print("\nSOURCE INFLUENCE AND SENSITIVITY AUDIT COMPLETED")
