import json
import numpy as np
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
df = pd.read_csv(root / "ctf_5deg_grain_boundary_segments.csv")
angles = df["misorientation_degrees"]
bins = [
    ("below_2", angles < 2),
    ("2_to_5", (angles >= 2) & (angles < 5)),
    ("5_to_10", (angles >= 5) & (angles < 10)),
    ("10_to_15", (angles >= 10) & (angles < 15)),
    ("15_to_30", (angles >= 15) & (angles < 30)),
    ("30_to_45", (angles >= 30) & (angles < 45)),
    ("45_and_above", angles >= 45),
]
print("=" * 65)
print("5-DEGREE SEGMENTATION / BOUNDARY CONSISTENCY AUDIT")
print("=" * 65)
print("\nBoundary angle distribution:")
for name, mask in bins:
    count = int(mask.sum())
    print(f"{name:20s} {count:7d} ({100*count/len(df):.3f}%)")
print("\nOverall statistics:")
print("Count:", len(df))
print("Minimum:", angles.min())
print("Maximum:", angles.max())
print("Mean:", angles.mean())
print("Median:", angles.median())
print("Below 5 degrees:", int((angles < 5).sum()))
low = df[angles < 5].copy()
if len(low):
    print("\nUnexpected sub-5-degree boundary segments:")
    print(low.nsmallest(30, "misorientation_degrees").to_string(index=False))
pair_summary = (
    df.groupby(["grain_1", "grain_2"])
    .agg(
        segments=("misorientation_degrees", "size"),
        mean_angle=("misorientation_degrees", "mean"),
        min_angle=("misorientation_degrees", "min"),
        max_angle=("misorientation_degrees", "max")
    )
    .reset_index()
)
print("\nUnique neighbouring pairs:", len(pair_summary))
print("Pairs with minimum angle below 5 degrees:",
      int((pair_summary["min_angle"] < 5).sum()))
print("Pairs with mean angle below 5 degrees:",
      int((pair_summary["mean_angle"] < 5).sum()))
pair_summary.to_csv(
    root / "ctf_5deg_boundary_pair_consistency.csv",
    index=False
)
report = {
    "boundary_segments": int(len(df)),
    "minimum_angle": float(angles.min()),
    "maximum_angle": float(angles.max()),
    "mean_angle": float(angles.mean()),
    "median_angle": float(angles.median()),
    "segments_below_5_degrees": int((angles < 5).sum()),
    "pairs_below_5_degrees": int(
        (pair_summary["min_angle"] < 5).sum()
    ),
    "pairs_mean_below_5_degrees": int(
        (pair_summary["mean_angle"] < 5).sum()
    ),
    "interpretation": (
        "Requires investigation if distinct neighbouring labels "
        "have boundary misorientation below the segmentation threshold."
    )
}
(root / "ctf_5deg_boundary_consistency_audit.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("\nAudit saved.")
