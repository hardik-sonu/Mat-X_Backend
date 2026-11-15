import json
import numpy as np
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
geometry = pd.read_csv(root / "ctf_grain_geometry_5deg.csv")
labels = pd.read_csv(root / "ctf_grain_pixels_5deg.csv")
boundaries = pd.read_csv(root / "ctf_5deg_grain_boundary_segments.csv")
neighbours = pd.read_csv(root / "ctf_5deg_grain_neighbour_statistics.csv")
# ---------------------------------------------------------
# 1. Validate source tables
# ---------------------------------------------------------
labels = labels.sort_values(["Y", "X"]).reset_index(drop=True)
nx = labels["X"].nunique()
ny = labels["Y"].nunique()
if len(labels) != nx * ny:
    raise RuntimeError("Incomplete grain-label grid.")
grain = labels["grain_id"].to_numpy(dtype=int).reshape(ny, nx)
ctf_path = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\EBSD.ctf"
)
with ctf_path.open("r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
header_index = next(
    i for i, line in enumerate(lines)
    if line.split()[:3] == ["Phase", "X", "Y"]
)
columns = lines[header_index].split()
data = pd.read_csv(
    ctf_path,
    sep=r"\s+",
    skiprows=header_index + 1,
    names=columns,
    header=None,
    engine="python"
).sort_values(["Y", "X"]).reset_index(drop=True)
if len(data) != len(labels):
    raise RuntimeError("CTF and grain-label row counts differ.")
if not np.allclose(
    data[["X", "Y"]].to_numpy(),
    labels[["X", "Y"]].to_numpy()
):
    raise RuntimeError("CTF and segmentation coordinates differ.")
valid = (
    (data["Phase"].to_numpy().reshape(ny, nx) == 1)
    & (data["Error"].to_numpy().reshape(ny, nx) == 0)
)
if not np.array_equal(valid, grain > 0):
    raise RuntimeError("Valid-pixel mask does not match grain labels.")
x_values = np.sort(data["X"].unique())
y_values = np.sort(data["Y"].unique())
dx = float(np.median(np.diff(x_values)))
dy = float(np.median(np.diff(y_values)))
if not np.isclose(dx, dy):
    print("WARNING: Anisotropic pixel spacing detected.")
print("Grid:", nx, "x", ny)
print("Valid pixels:", int(valid.sum()))
print("Pixel spacing:", dx, dy)
# ---------------------------------------------------------
# 2. Grain perimeter and boundary exposure
# ---------------------------------------------------------
max_id = int(grain.max())
perimeter_length = np.zeros(max_id + 1, dtype=float)
internal_length = np.zeros(max_id + 1, dtype=float)
external_length = np.zeros(max_id + 1, dtype=float)
padded = np.pad(grain, 1, mode="constant", constant_values=0)
center = padded[1:-1, 1:-1]
directions = [
    (padded[1:-1, :-2], dy),  # left/right edges: length dy
    (padded[1:-1, 2:], dy),
    (padded[:-2, 1:-1], dx),  # top/bottom edges: length dx
    (padded[2:, 1:-1], dx)
]
for adjacent, edge_length in directions:
    exposed = (center > 0) & (adjacent != center)
    ids = center[exposed]
    perimeter_length += np.bincount(
        ids,
        weights=np.full(len(ids), edge_length),
        minlength=max_id + 1
    )[:max_id + 1]
    internal = exposed & (adjacent > 0)
    internal_ids = center[internal]
    internal_length += np.bincount(
        internal_ids,
        weights=np.full(len(internal_ids), edge_length),
        minlength=max_id + 1
    )[:max_id + 1]
    external = exposed & (adjacent == 0)
    external_ids = center[external]
    external_length += np.bincount(
        external_ids,
        weights=np.full(len(external_ids), edge_length),
        minlength=max_id + 1
    )[:max_id + 1]
# ---------------------------------------------------------
# 3. Assemble geometry descriptors
# ---------------------------------------------------------
features = geometry.copy()
ids = features["grain_id"].to_numpy(dtype=int)
features["perimeter_ctf_coordinate_units"] = perimeter_length[ids]
features["internal_boundary_length_ctf_units"] = internal_length[ids]
features["unindexed_or_map_edge_length_ctf_units"] = external_length[ids]
features["bbox_width_pixels"] = (
    np.rint((features["x_max"] - features["x_min"]) / dx).astype(int) + 1
)
features["bbox_height_pixels"] = (
    np.rint((features["y_max"] - features["y_min"]) / dy).astype(int) + 1
)
area = features["area_ctf_coordinate_units_squared"].to_numpy()
perimeter = features["perimeter_ctf_coordinate_units"].to_numpy()
features["circularity"] = np.where(
    perimeter > 0,
    4 * np.pi * area / perimeter**2,
    np.nan
)
features["aspect_ratio_bbox"] = (
    features["bbox_width_pixels"]
    / features["bbox_height_pixels"].replace(0, np.nan)
)
features["boundary_exposure_fraction"] = np.where(
    perimeter > 0,
    external_length[ids] / perimeter,
    np.nan
)
# ---------------------------------------------------------
# 4. Add grain-neighbour statistics
# ---------------------------------------------------------
features = features.merge(
    neighbours,
    on="grain_id",
    how="left",
    validate="one_to_one"
)
# ---------------------------------------------------------
# 5. Aggregate grain boundary misorientation
# ---------------------------------------------------------
side_a = boundaries[
    ["grain_1", "misorientation_degrees"]
].rename(columns={"grain_1": "grain_id"})
side_b = boundaries[
    ["grain_2", "misorientation_degrees"]
].rename(columns={"grain_2": "grain_id"})
records = pd.concat([side_a, side_b], ignore_index=True)
records["high_angle"] = (
    records["misorientation_degrees"] >= 15
).astype(int)
records["low_angle"] = (
    (records["misorientation_degrees"] >= 5)
    & (records["misorientation_degrees"] < 15)
).astype(int)
boundary_features = (
    records.groupby("grain_id")
    .agg(
        boundary_segment_count=("misorientation_degrees", "size"),
        mean_boundary_misorientation=("misorientation_degrees", "mean"),
        median_boundary_misorientation=("misorientation_degrees", "median"),
        std_boundary_misorientation=("misorientation_degrees", "std"),
        minimum_boundary_misorientation=("misorientation_degrees", "min"),
        maximum_boundary_misorientation=("misorientation_degrees", "max"),
        high_angle_boundary_count=("high_angle", "sum"),
        low_angle_boundary_count=("low_angle", "sum")
    )
    .reset_index()
)
features = features.merge(
    boundary_features,
    on="grain_id",
    how="left",
    validate="one_to_one"
)
features["high_angle_boundary_fraction"] = (
    features["high_angle_boundary_count"]
    / features["boundary_segment_count"]
)
features["low_angle_boundary_fraction"] = (
    features["low_angle_boundary_count"]
    / features["boundary_segment_count"]
)
# ---------------------------------------------------------
# 6. Scientific and data-integrity checks
# ---------------------------------------------------------
if len(features) != len(geometry):
    raise RuntimeError("Grain count changed.")
if features["grain_id"].duplicated().any():
    raise RuntimeError("Duplicate grain IDs.")
if features["boundary_segment_count"].isna().any():
    raise RuntimeError("Grains without boundary records detected.")
fraction_sum = (
    features["high_angle_boundary_fraction"]
    + features["low_angle_boundary_fraction"]
)
if not np.allclose(fraction_sum, 1.0):
    raise RuntimeError("Boundary fractions do not sum to one.")
if (features["perimeter_ctf_coordinate_units"] <= 0).any():
    raise RuntimeError("Non-positive grain perimeter detected.")
if not np.isfinite(
    features["circularity"].dropna()
).all():
    raise RuntimeError("Invalid circularity values.")
# ---------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------
output_csv = root / "ctf_5deg_grain_level_features.csv"
output_json = root / "ctf_5deg_grain_level_features_report.json"
features.to_csv(output_csv, index=False)
summary = {
    "segmentation_threshold_degrees": 5,
    "grain_count": int(len(features)),
    "valid_pixels": int(valid.sum()),
    "pixel_spacing_ctf_coordinate_units": {
        "x": dx,
        "y": dy
    },
    "feature_count": int(len(features.columns)),
    "features": list(features.columns),
    "grain_size": {
        "mean_equivalent_diameter": float(
            features["equivalent_diameter_ctf_coordinate_units"].mean()
        ),
        "median_equivalent_diameter": float(
            features["equivalent_diameter_ctf_coordinate_units"].median()
        )
    },
    "neighbours": {
        "mean": float(features["neighbour_count"].mean()),
        "median": float(features["neighbour_count"].median()),
        "maximum": int(features["neighbour_count"].max())
    },
    "boundary": {
        "mean_grain_boundary_misorientation": float(
            features["mean_boundary_misorientation"].mean()
        ),
        "mean_high_angle_boundary_fraction": float(
            features["high_angle_boundary_fraction"].mean()
        )
    },
    "shape": {
        "median_circularity": float(features["circularity"].median()),
        "median_bbox_aspect_ratio": float(
            features["aspect_ratio_bbox"].median()
        )
    },
    "perimeter": {
        "mean": float(features["perimeter_ctf_coordinate_units"].mean()),
        "median": float(features["perimeter_ctf_coordinate_units"].median())
    },
    "quality": {
        "missing_boundary_feature_rows": int(
            features["boundary_segment_count"].isna().sum()
        ),
        "duplicate_grain_ids": int(
            features["grain_id"].duplicated().sum()
        )
    },
    "limitations": [
        "Segmentation remains provisional at 5 degrees.",
        "CTF coordinate units are not confirmed as micrometres.",
        "Circularity and aspect ratio are pixel-grid approximations.",
        "Euler convention and physical reference frame remain subject to source verification.",
        "Boundary exposure combines map-edge and unindexed-neighbour exposure.",
        "These descriptors are not validated material-property predictors."
    ]
}
output_json.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("\n" + "=" * 70)
print("GRAIN-LEVEL FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nOutput files:")
print(output_csv)
print(output_json)
print("\nFirst 10 grains:")
print(
    features[
        [
            "grain_id",
            "pixel_count",
            "equivalent_diameter_ctf_coordinate_units",
            "circularity",
            "aspect_ratio_bbox",
            "neighbour_count",
            "mean_boundary_misorientation",
            "high_angle_boundary_fraction"
        ]
    ].head(10).to_string(index=False)
)
print("\nOriginal dataset and previous outputs unchanged.")
