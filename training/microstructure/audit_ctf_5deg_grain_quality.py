from pathlib import Path
import pandas as pd
import numpy as np
import json
out = Path("training_results/microstructure/abaqus_deep_audit")
pixels = pd.read_csv(out / "ctf_grain_pixels_5deg.csv")
grains = pd.read_csv(out / "ctf_grain_geometry_5deg.csv")
nx = pixels["X"].nunique()
ny = pixels["Y"].nunique()
if len(pixels) != nx * ny:
    raise RuntimeError("Pixel grid is incomplete.")
if pixels.duplicated(["X", "Y"]).any():
    raise RuntimeError("Duplicate coordinates found.")
small = grains[grains["pixel_count"] < 5].copy()
# A grain touches the measurement-map boundary if its bounding
# box reaches any outer coordinate.
xmin, xmax = pixels["X"].min(), pixels["X"].max()
ymin, ymax = pixels["Y"].min(), pixels["Y"].max()
grains["touches_map_edge"] = (
    (grains["x_min"] == xmin) |
    (grains["x_max"] == xmax) |
    (grains["y_min"] == ymin) |
    (grains["y_max"] == ymax)
)
small["touches_map_edge"] = (
    (small["x_min"] == xmin) |
    (small["x_max"] == xmax) |
    (small["y_min"] == ymin) |
    (small["y_max"] == ymax)
)
size_bins = {
    "1 pixel": int((grains["pixel_count"] == 1).sum()),
    "2-4 pixels": int(grains["pixel_count"].between(2, 4).sum()),
    "5-10 pixels": int(grains["pixel_count"].between(5, 10).sum()),
    "11-25 pixels": int(grains["pixel_count"].between(11, 25).sum()),
    "26-100 pixels": int(grains["pixel_count"].between(26, 100).sum()),
    "over 100 pixels": int((grains["pixel_count"] > 100).sum())
}
summary = {
    "threshold_degrees": 5,
    "grid": [int(nx), int(ny)],
    "grain_count": int(len(grains)),
    "indexed_pixels": int(grains["pixel_count"].sum()),
    "grain_size_bins": size_bins,
    "small_grain_count_under_5_pixels": int(len(small)),
    "pixels_in_grains_under_5": int(small["pixel_count"].sum()),
    "small_grains_touching_map_edge": int(small["touches_map_edge"].sum()),
    "all_grains_touching_map_edge": int(grains["touches_map_edge"].sum()),
    "grain_size_percentiles_pixels": {
        str(p): float(grains["pixel_count"].quantile(p / 100))
        for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]
    }
}
small.to_csv(out / "ctf_5deg_small_grains_review.csv", index=False)
grains.to_csv(out / "ctf_5deg_grain_edge_audit.csv", index=False)
(out / "ctf_5deg_grain_quality_audit.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 70)
print("5-DEGREE GRAIN QUALITY AUDIT")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nSMALL GRAIN RECORDS:")
print(small.to_string(index=False))
print("\nSaved reports in:", out)
print("Original CTF unchanged.")
