from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(
    r"training_results\uncertainty_calibration_v1"
    r"\calibration_residual_diagnostics"
)
FILE = ROOT / "tensile_strength_individual_calibration_residuals.csv"
df = pd.read_csv(FILE)
nims = df.loc[
    df["grade_id"].astype(str) == "NIMS_low_alloy_other",
    "absolute_residual"
].dropna().to_numpy(dtype=float)
if len(nims) != 81:
    raise SystemExit(f"Expected 81 NIMS residuals, found {len(nims)}")
rng = np.random.default_rng(42)
iterations = 20000
sample_sizes = [1, 2, 3, 4, 7, 10, 20, 40, 81]
print("=" * 85)
print("NIMS WITHIN-GROUP MAXIMUM RESIDUAL SIZE SENSITIVITY")
print("=" * 85)
print("NIMS residual count:", len(nims))
print("NIMS mean absolute residual:", round(float(nims.mean()), 4))
print("NIMS median absolute residual:", round(float(np.median(nims)), 4))
print("NIMS observed maximum:", round(float(nims.max()), 4))
print("Monte Carlo iterations:", iterations)
rows = []
for size in sample_sizes:
    maxima = np.empty(iterations)
    for i in range(iterations):
        sample = rng.choice(nims, size=size, replace=False)
        maxima[i] = sample.max()
    rows.append({
        "sample_size": size,
        "mean_sample_max": maxima.mean(),
        "median_sample_max": np.median(maxima),
        "p90_sample_max": np.quantile(maxima, 0.90),
        "p95_sample_max": np.quantile(maxima, 0.95),
        "maximum_possible": maxima.max(),
    })
result = pd.DataFrame(rows)
print("\nSIMULATED MAXIMUM RESIDUAL BY SAMPLE SIZE:")
print(result.round(3).to_string(index=False))
print("\nINTERPRETATION:")
print(
    "This simulation estimates the effect of taking a maximum over "
    "different numbers of records drawn from the same observed NIMS "
    "residual population. It does not establish exchangeability "
    "between grades or validate a new calibration method."
)
print("\nSIMULATION COMPLETE — NO DATA OR MODELS MODIFIED.")
