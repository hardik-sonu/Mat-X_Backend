import json
import numpy as np
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
f = pd.read_csv(root / "ctf_5deg_grain_level_features.csv")
labels = pd.read_csv(root / "ctf_grain_pixels_5deg.csv")
boundaries = pd.read_csv(root / "ctf_5deg_grain_boundary_segments.csv")
dx = 1.1
dy = 1.1
pixel_area = dx * dy
# ---------------------------------------------------------
# 1. Area conservation
# ---------------------------------------------------------
valid_pixels = int((labels["grain_id"] > 0).sum())
expected_area = valid_pixels * pixel_area
measured_area = float(f["area_ctf_coordinate_units_squared"].sum())
area_error = measured_area - expected_area
# ---------------------------------------------------------
# 2. Perimeter accounting
# ---------------------------------------------------------
total_perimeter = float(
    f["perimeter_ctf_coordinate_units"].sum()
)
total_internal = float(
    f["internal_boundary_length_ctf_units"].sum()
)
total_external = float(
    f["unindexed_or_map_edge_length_ctf_units"].sum()
)
perimeter_identity_error = (
    total_perimeter - total_internal - total_external
)
# Each internal interface contributes to both grains.
# Therefore total perimeter should equal the sum of all
# internal and external grain-side edge contributions.
# ---------------------------------------------------------
# 3. Shape quality
# ---------------------------------------------------------
circularity = f["circularity"]
circularity_invalid = int(
    ((circularity < 0) | (circularity > 1.000001)).sum()
)
# ---------------------------------------------------------
# 4. Grain size distribution
# ---------------------------------------------------------
size = f["pixel_count"]
size_bins = {
    "1_pixel": int((size == 1).sum()),
    "2_to_4_pixels": int(size.between(2, 4).sum()),
    "5_to_10_pixels": int(size.between(5, 10).sum()),
    "11_to_25_pixels": int(size.between(11, 25).sum()),
    "26_to_100_pixels": int(size.between(26, 100).sum()),
    "over_100_pixels": int((size > 100).sum())
}
# ---------------------------------------------------------
# 5. Boundary exposure
# ---------------------------------------------------------
exposure = f["boundary_exposure_fraction"]
exposure_summary = {
    "mean": float(exposure.mean()),
    "median": float(exposure.median()),
    "p90": float(exposure.quantile(.90)),
    "maximum": float(exposure.max()),
    "grains_with_any_exposure": int((exposure > 0).sum()),
    "grains_over_25_percent": int((exposure > .25).sum()),
    "grains_over_50_percent": int((exposure > .50).sum())
}
# ---------------------------------------------------------
# 6. Boundary network consistency
# ---------------------------------------------------------
boundary_segments = len(boundaries)
grain_side_segments = int(
    f["boundary_segment_count"].sum()
)
expected_grain_side_segments = 2 * boundary_segments
# ---------------------------------------------------------
# 7. Neighbour and boundary distributions
# ---------------------------------------------------------
neighbour_bins = {
    str(n): int((f["neighbour_count"] == n).sum())
    for n in sorted(f["neighbour_count"].dropna().unique())
}
summary = {
    "grain_count": int(len(f)),
    "valid_pixels": valid_pixels,
    "area_conservation": {
        "expected_area": expected_area,
        "measured_area": measured_area,
        "difference": area_error,
        "passed": bool(np.isclose(
            expected_area, measured_area, rtol=1e-10, atol=1e-8
        ))
    },
    "perimeter_accounting": {
        "total_grain_perimeter": total_perimeter,
        "internal_boundary_contributions": total_internal,
        "external_boundary_contributions": total_external,
        "identity_difference": perimeter_identity_error
    },
    "circularity": {
        "minimum": float(circularity.min()),
        "median": float(circularity.median()),
        "maximum": float(circularity.max()),
        "invalid_count": circularity_invalid
    },
    "grain_size_bins": size_bins,
    "grain_size_percentiles_pixels": {
        str(p): float(size.quantile(p / 100))
        for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]
    },
    "boundary_exposure": exposure_summary,
    "boundary_network": {
        "boundary_pixel_segments": int(boundary_segments),
        "sum_of_grain_side_segments": grain_side_segments,
        "expected_grain_side_segments": expected_grain_side_segments,
        "passed": grain_side_segments == expected_grain_side_segments
    },
    "neighbour_distribution": neighbour_bins,
    "limitations": [
        "Boundary exposure combines map-edge and unindexed-neighbour edges.",
        "Circularity is based on a pixel-grid perimeter.",
        "Grain segmentation remains provisional.",
        "CTF coordinate units are not confirmed as micrometres."
    ]
}
out = root / "ctf_5deg_grain_feature_quality_audit.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("=" * 70)
print("GRAIN FEATURE QUALITY AUDIT")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nSmallest 15 grains:")
print(
    f.nsmallest(15, "pixel_count")[
        ["grain_id", "pixel_count", "circularity",
         "neighbour_count", "boundary_exposure_fraction"]
    ].to_string(index=False)
)
print("\nLargest 10 grains:")
print(
    f.nlargest(10, "pixel_count")[
        ["grain_id", "pixel_count", "circularity",
         "neighbour_count", "boundary_exposure_fraction"]
    ].to_string(index=False)
)
print("\nSaved:", out)
