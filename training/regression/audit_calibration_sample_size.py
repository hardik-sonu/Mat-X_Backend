from pathlib import Path
import math
import numpy as np
import pandas as pd
ROOT = Path("training_results/uncertainty_calibration_v1")
files = [
    ROOT / "tensile_strength_baseline_calibration_grade_scores.csv",
    ROOT / "yield_strength_baseline_calibration_grade_scores.csv",
]
for path in files:
    print("\n" + "=" * 72)
    print(path.name)
    print("=" * 72)
    if not path.exists():
        print("FILE NOT FOUND")
        continue
    df = pd.read_csv(path)
    col = "maximum_absolute_residual"
    scores = pd.to_numeric(df[col], errors="coerce").dropna()
    scores = scores[np.isfinite(scores)].to_numpy()
    n = len(scores)
    print("Calibration grades:", n)
    print("Unique score count:", len(np.unique(scores)))
    print("Minimum score:", float(np.min(scores)))
    print("Median score:", float(np.median(scores)))
    print("Maximum score:", float(np.max(scores)))
    ordered = np.sort(scores)
    for confidence in (0.90, 0.95):
        rank = math.ceil((n + 1) * confidence)
        q = float("inf") if rank > n else float(ordered[rank - 1])
        print(
            f"{confidence:.0%}: rank={rank}, "
            f"quantile={q}, "
            f"finite_interval={math.isfinite(q)}"
        )
print("\nCALIBRATION SAMPLE AUDIT COMPLETED")
