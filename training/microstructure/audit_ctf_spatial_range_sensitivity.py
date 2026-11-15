import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
ROOT = Path("training_results/microstructure/abaqus_deep_audit")
INPUT = ROOT / "ctf_5deg_matx_grain_features_quality_annotated.csv"
FEATURES = [
    "pixel_count",
    "equivalent_diameter_ctf_coordinate_units",
    "circularity",
    "aspect_ratio_bbox",
    "neighbour_count",
    "mean_boundary_misorientation",
    "high_angle_boundary_fraction",
    "boundary_exposure_fraction"
]
BIN_WIDTHS = [5.0, 10.0, 15.0, 20.0]
SILL_THRESHOLDS = [0.90, 0.95, 1.00]
MIN_PAIRS = 30
CONSECUTIVE_BINS = 3
df = pd.read_csv(INPUT)
subsets = {
    "ALL_GRAINS": df.copy(),
    "GEOMETRY_SCREEN_PASS": df.loc[
        df["geometry_screen_pass"].astype(bool)
    ].copy()
}
results = []
variogram_rows = []
for subset_name, frame in subsets.items():
    frame = frame.reset_index(drop=True)
    coordinates = frame[
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    for feature in FEATURES:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        coordinates_valid = coordinates[valid]
        values = values[valid]
        if len(values) < 10 or np.var(values) <= 1e-15:
            continue
        distances = pdist(coordinates_valid)
        squared_differences = pdist(
            values.reshape(-1, 1),
            metric="sqeuclidean"
        )
        variance = float(np.var(values))
        max_distance = float(distances.max())
        for width in BIN_WIDTHS:
            edges = np.arange(
                0,
                np.ceil(max_distance / width) * width + width,
                width
            )
            if edges[-1] <= max_distance:
                edges = np.append(edges, edges[-1] + width)
            bin_ids = np.digitize(distances, edges) - 1
            bins = []
            for b in range(len(edges) - 1):
                mask = bin_ids == b
                count = int(mask.sum())
                if count == 0:
                    continue
                gamma = float(
                    0.5 * squared_differences[mask].mean()
                )
                normalized = gamma / variance
                record = {
                    "subset": subset_name,
                    "feature": feature,
                    "bin_width": width,
                    "distance_min": float(edges[b]),
                    "distance_max": float(edges[b + 1]),
                    "distance_midpoint": float(
                        (edges[b] + edges[b + 1]) / 2
                    ),
                    "pair_count": count,
                    "normalized_semivariance": normalized
                }
                bins.append(record)
                variogram_rows.append(record)
            for sill in SILL_THRESHOLDS:
                estimate = None
                status = "SILL_NOT_REACHED"
                for i in range(
                    len(bins) - CONSECUTIVE_BINS + 1
                ):
                    window = bins[
                        i:i + CONSECUTIVE_BINS
                    ]
                    # Require consecutive distance bins.
                    consecutive = all(
                        np.isclose(
                            window[j]["distance_min"],
                            window[j - 1]["distance_max"]
                        )
                        for j in range(1, len(window))
                    )
                    enough_pairs = all(
                        item["pair_count"] >= MIN_PAIRS
                        for item in window
                    )
                    reached = all(
                        item["normalized_semivariance"] >= sill
                        for item in window
                    )
                    if consecutive and enough_pairs and reached:
                        estimate = window[0]["distance_min"]
                        if estimate == 0:
                            status = "BELOW_BIN_RESOLUTION"
                        else:
                            status = "APPROXIMATE_SILL_DISTANCE"
                        break
                results.append({
                    "subset": subset_name,
                    "feature": feature,
                    "bin_width": width,
                    "sill_threshold": sill,
                    "estimated_distance": estimate,
                    "status": status,
                    "minimum_pairs_per_bin": MIN_PAIRS,
                    "consecutive_bins_required": CONSECUTIVE_BINS
                })
result_df = pd.DataFrame(results)
variogram_df = pd.DataFrame(variogram_rows)
result_path = ROOT / "ctf_5deg_spatial_range_sensitivity.csv"
variogram_path = ROOT / "ctf_5deg_spatial_range_sensitivity_variograms.csv"
report_path = ROOT / "ctf_5deg_spatial_range_sensitivity_report.json"
result_df.to_csv(result_path, index=False)
variogram_df.to_csv(variogram_path, index=False)
summary = {
    "method": "Empirical omnidirectional semivariogram sensitivity",
    "bin_widths": BIN_WIDTHS,
    "sill_thresholds": SILL_THRESHOLDS,
    "minimum_pairs_per_bin": MIN_PAIRS,
    "consecutive_bins_required": CONSECUTIVE_BINS,
    "distance_unit": "CTF coordinate units; physical unit unconfirmed",
    "results": results,
    "interpretation": [
        "BELOW_BIN_RESOLUTION means the sill criterion is already met from the first bin; it does not mean zero correlation distance.",
        "APPROXIMATE_SILL_DISTANCE is a descriptive threshold crossing, not a validated independence distance.",
        "SILL_NOT_REACHED means the selected criterion was not met.",
        "Large sensitivity to bin width or sill threshold indicates an unstable range estimate.",
        "Spatial ranges from this one map cannot replace independent specimen validation."
    ]
}
report_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 76)
print("SPATIAL RANGE SENSITIVITY AUDIT")
print("=" * 76)
print("\nSensitivity results:")
print(result_df.to_string(index=False))
print("\nStatus counts:")
print(
    result_df.groupby(
        ["subset", "status"]
    ).size().to_string()
)
print("\nSaved:")
print(result_path)
print(variogram_path)
print(report_path)
