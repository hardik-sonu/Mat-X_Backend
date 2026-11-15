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
BIN_WIDTH = 10.0
MIN_PAIRS = 30
df = pd.read_csv(INPUT)
subsets = {
    "ALL_GRAINS": df.copy(),
    "GEOMETRY_SCREEN_PASS": df.loc[
        df["geometry_screen_pass"].astype(bool)
    ].copy()
}
all_results = []
range_results = []
for subset_name, frame in subsets.items():
    frame = frame.reset_index(drop=True)
    coordinates = frame[
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    distances = pdist(coordinates, metric="euclidean")
    max_distance = float(distances.max())
    edges = np.arange(
        0,
        np.ceil(max_distance / BIN_WIDTH) * BIN_WIDTH
        + BIN_WIDTH,
        BIN_WIDTH
    )
    if edges[-1] <= max_distance:
        edges = np.append(edges, edges[-1] + BIN_WIDTH)
    bin_ids = np.digitize(distances, edges) - 1
    bin_count = len(edges) - 1
    for feature in FEATURES:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.all():
            valid_frame = frame.loc[valid].reset_index(drop=True)
            valid_coordinates = valid_frame[
                ["centroid_x", "centroid_y"]
            ].to_numpy(dtype=float)
            valid_distances = pdist(valid_coordinates)
            valid_edges = np.arange(
                0,
                np.ceil(valid_distances.max() / BIN_WIDTH)
                * BIN_WIDTH + BIN_WIDTH,
                BIN_WIDTH
            )
            if valid_edges[-1] <= valid_distances.max():
                valid_edges = np.append(
                    valid_edges,
                    valid_edges[-1] + BIN_WIDTH
                )
            current_edges = valid_edges
            current_distances = valid_distances
            current_values = values[valid]
        else:
            current_edges = edges
            current_distances = distances
            current_values = values
        n = len(current_values)
        if n < 10 or np.var(current_values) <= 1e-15:
            range_results.append({
                "subset": subset_name,
                "feature": feature,
                "n": n,
                "estimated_range": None,
                "status": "INSUFFICIENT_OR_CONSTANT"
            })
            continue
        squared_differences = pdist(
            current_values.reshape(-1, 1),
            metric="sqeuclidean"
        )
        current_bin_ids = np.digitize(
            current_distances,
            current_edges
        ) - 1
        variance = float(np.var(current_values))
        rows = []
        for b in range(len(current_edges) - 1):
            mask = current_bin_ids == b
            pair_count = int(mask.sum())
            if pair_count == 0:
                continue
            semivariance = float(
                0.5 * squared_differences[mask].mean()
            )
            normalized_semivariance = semivariance / variance
            rows.append({
                "subset": subset_name,
                "feature": feature,
                "distance_min": float(current_edges[b]),
                "distance_max": float(current_edges[b + 1]),
                "distance_midpoint": float(
                    (current_edges[b] + current_edges[b + 1]) / 2
                ),
                "pair_count": pair_count,
                "semivariance": semivariance,
                "normalized_semivariance": normalized_semivariance,
                "correlation_proxy": 1.0 - normalized_semivariance
            })
        all_results.extend(rows)
        # A practical descriptive range:
        # first distance bin in a run of three consecutive populated
        # bins with normalized semivariance >= 0.95.
        qualifying = []
        for i in range(len(rows) - 2):
            three = rows[i:i + 3]
            enough_pairs = all(
                item["pair_count"] >= MIN_PAIRS
                for item in three
            )
            plateau = all(
                item["normalized_semivariance"] >= 0.95
                for item in three
            )
            if enough_pairs and plateau:
                qualifying.append(three[0]["distance_min"])
                break
        range_results.append({
            "subset": subset_name,
            "feature": feature,
            "n": n,
            "estimated_range": (
                float(qualifying[0]) if qualifying else None
            ),
            "status": (
                "APPROXIMATE_RANGE_FOUND"
                if qualifying
                else "RANGE_NOT_ESTABLISHED"
            )
        })
variogram = pd.DataFrame(all_results)
ranges = pd.DataFrame(range_results)
variogram_path = ROOT / "ctf_5deg_empirical_semivariograms.csv"
range_path = ROOT / "ctf_5deg_spatial_correlation_ranges.csv"
report_path = ROOT / "ctf_5deg_spatial_correlation_range_report.json"
variogram.to_csv(variogram_path, index=False)
ranges.to_csv(range_path, index=False)
summary = {
    "dataset": "EBSD.ctf",
    "method": "Empirical omnidirectional semivariogram",
    "distance_bin_width": BIN_WIDTH,
    "distance_unit": "CTF coordinate units; physical unit unconfirmed",
    "minimum_pairs_per_bin_for_range": MIN_PAIRS,
    "range_rule": (
        "First of three consecutive populated distance bins "
        "with normalized semivariance >= 0.95"
    ),
    "results": ranges.to_dict(orient="records"),
    "limitations": [
        "A single EBSD map is being analysed.",
        "The semivariogram assumes approximate spatial stationarity.",
        "An omnidirectional analysis does not test directional anisotropy.",
        "Estimated ranges are descriptive and depend on bin width and rule.",
        "A range estimate is not proof that observations beyond it are independent.",
        "CTF coordinate units are not confirmed as micrometres.",
        "Independent specimen-level validation remains necessary."
    ]
}
report_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 72)
print("MAT-X EMPIRICAL SPATIAL CORRELATION RANGE AUDIT")
print("=" * 72)
print("\nEstimated ranges:")
print(ranges.to_string(index=False))
print("\nFirst variogram bins:")
print(
    variogram.groupby(
        ["subset", "feature"],
        sort=False
    ).head(5).to_string(index=False)
)
print("\nSaved:")
print(variogram_path)
print(range_path)
print(report_path)
