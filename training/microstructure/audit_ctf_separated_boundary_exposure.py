from pathlib import Path
import json
import numpy as np
import pandas as pd
PROJECT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATASET = (
    Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
    / "Grain orientation and cyclic behavior"
)
OUT = (
    PROJECT
    / "training_results"
    / "microstructure"
    / "abaqus_deep_audit"
)
OUT.mkdir(parents=True, exist_ok=True)
ctf_path = DATASET / "Dataset" / "EBSD.ctf"
labels_path = OUT / "ctf_grain_pixels_5deg.csv"
if not ctf_path.exists():
    raise FileNotFoundError(ctf_path)
if not labels_path.exists():
    raise FileNotFoundError(labels_path)
# ---------------------------------------------------------
# 1. Detect the CTF measurement header robustly
# ---------------------------------------------------------
header_index = None
with ctf_path.open("r", encoding="utf-8-sig", errors="replace") as f:
    for i, line in enumerate(f):
        tokens = line.strip().split()
        if (
            len(tokens) >= 11
            and tokens[0] == "Phase"
            and tokens[1:5] == ["X", "Y", "Bands", "Error"]
            and tokens[5:8] == ["Euler1", "Euler2", "Euler3"]
        ):
            header_index = i
            break
if header_index is None:
    raise RuntimeError("Could not identify the CTF measurement header.")
ctf = pd.read_csv(
    ctf_path,
    sep=r"\s+",
    skiprows=header_index,
    engine="python"
)
required = {"Phase", "X", "Y", "Error"}
missing = required - set(ctf.columns)
if missing:
    raise RuntimeError(f"CTF columns missing: {sorted(missing)}")
ctf = ctf.sort_values(["Y", "X"]).reset_index(drop=True)
x_values = np.sort(ctf["X"].unique())
y_values = np.sort(ctf["Y"].unique())
height = len(y_values)
width = len(x_values)
if height * width != len(ctf):
    raise RuntimeError(
        f"Incomplete CTF grid: {height} x {width} != {len(ctf)}"
    )
if ctf.duplicated(["X", "Y"]).any():
    raise RuntimeError("Duplicate CTF grid coordinates detected.")
phase = ctf["Phase"].to_numpy().reshape(height, width)
error = ctf["Error"].to_numpy().reshape(height, width)
# Only indexed FCC phase-1 points are valid.
valid_grid = (phase == 1) & (error == 0)
# ---------------------------------------------------------
# 2. Load the existing 5-degree grain labels
# ---------------------------------------------------------
labels_df = pd.read_csv(labels_path)
required_label_columns = {"X", "Y", "grain_id"}
missing = required_label_columns - set(labels_df.columns)
if missing:
    raise RuntimeError(
        f"Grain-label columns missing: {sorted(missing)}. "
        f"Found: {list(labels_df.columns)}"
    )
labels_df = labels_df.sort_values(["Y", "X"]).reset_index(drop=True)
if len(labels_df) != len(ctf):
    raise RuntimeError(
        f"Label rows {len(labels_df)} != CTF rows {len(ctf)}"
    )
if not np.allclose(
    labels_df["X"].to_numpy(),
    ctf["X"].to_numpy()
):
    raise RuntimeError("X coordinates do not align.")
if not np.allclose(
    labels_df["Y"].to_numpy(),
    ctf["Y"].to_numpy()
):
    raise RuntimeError("Y coordinates do not align.")
labels = (
    labels_df["grain_id"]
    .to_numpy(dtype=np.int64)
    .reshape(height, width)
)
if int(valid_grid.sum()) != 73163:
    raise RuntimeError(
        f"Unexpected valid-pixel count: {int(valid_grid.sum())}"
    )
if not np.array_equal(valid_grid, labels > 0):
    raise RuntimeError(
        "Grain-label mask does not exactly match Phase==1 and Error==0."
    )
grain_ids = np.unique(labels[valid_grid])
if len(grain_ids) != 652:
    raise RuntimeError(f"Expected 652 grains; found {len(grain_ids)}.")
# ---------------------------------------------------------
# 3. Count perimeter sides by exposure type
# ---------------------------------------------------------
map_edge_sides = np.zeros((height, width), dtype=np.int8)
unindexed_sides = np.zeros((height, width), dtype=np.int8)
internal_sides = np.zeros((height, width), dtype=np.int8)
# Top edge
map_edge_sides[0, :] += valid_grid[0, :]
# Bottom edge
map_edge_sides[-1, :] += valid_grid[-1, :]
# Left edge
map_edge_sides[:, 0] += valid_grid[:, 0]
# Right edge
map_edge_sides[:, -1] += valid_grid[:, -1]
# Compare each valid pixel with its four in-map neighbours.
directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]
for dr, dc in directions:
    source_r0 = max(0, -dr)
    source_r1 = min(height, height - dr)
    source_c0 = max(0, -dc)
    source_c1 = min(width, width - dc)
    neighbour_r0 = source_r0 + dr
    neighbour_r1 = source_r1 + dr
    neighbour_c0 = source_c0 + dc
    neighbour_c1 = source_c1 + dc
    source_valid = valid_grid[
        source_r0:source_r1,
        source_c0:source_c1
    ]
    neighbour_valid = valid_grid[
        neighbour_r0:neighbour_r1,
        neighbour_c0:neighbour_c1
    ]
    source_labels = labels[
        source_r0:source_r1,
        source_c0:source_c1
    ]
    neighbour_labels = labels[
        neighbour_r0:neighbour_r1,
        neighbour_c0:neighbour_c1
    ]
    unindexed = (
        source_valid & (~neighbour_valid)
    )
    internal = (
        source_valid
        & neighbour_valid
        & (source_labels != neighbour_labels)
    )
    target_unindexed = unindexed_sides[
        source_r0:source_r1,
        source_c0:source_c1
    ]
    target_internal = internal_sides[
        source_r0:source_r1,
        source_c0:source_c1
    ]
    target_unindexed += unindexed.astype(np.int8)
    target_internal += internal.astype(np.int8)
# ---------------------------------------------------------
# 4. Aggregate side counts per grain
# ---------------------------------------------------------
max_grain_id = int(grain_ids.max())
def aggregate_by_grain(values):
    return np.bincount(
        labels[valid_grid],
        weights=values[valid_grid],
        minlength=max_grain_id + 1
    )
pixel_counts = aggregate_by_grain(
    np.ones((height, width), dtype=np.int8)
)
map_counts = aggregate_by_grain(map_edge_sides)
unindexed_counts = aggregate_by_grain(unindexed_sides)
internal_counts = aggregate_by_grain(internal_sides)
rows = []
for gid in grain_ids:
    gid = int(gid)
    pixels = int(pixel_counts[gid])
    map_sides = int(map_counts[gid])
    unindexed_side_count = int(unindexed_counts[gid])
    internal_side_count = int(internal_counts[gid])
    perimeter = (
        map_sides
        + unindexed_side_count
        + internal_side_count
    )
    rows.append({
        "grain_id": gid,
        "pixel_count": pixels,
        "map_edge_exposure_sides": map_sides,
        "unindexed_neighbour_exposure_sides": unindexed_side_count,
        "internal_grain_boundary_sides": internal_side_count,
        "total_perimeter_sides": perimeter,
        "map_edge_exposure_fraction": (
            map_sides / perimeter if perimeter else np.nan
        ),
        "unindexed_exposure_fraction": (
            unindexed_side_count / perimeter if perimeter else np.nan
        ),
        "internal_boundary_fraction": (
            internal_side_count / perimeter if perimeter else np.nan
        )
    })
result = pd.DataFrame(rows)
# ---------------------------------------------------------
# 5. Validate perimeter accounting and save
# ---------------------------------------------------------
identity = (
    result["total_perimeter_sides"]
    ==
    result[
        [
            "map_edge_exposure_sides",
            "unindexed_neighbour_exposure_sides",
            "internal_grain_boundary_sides"
        ]
    ].sum(axis=1)
)
if not identity.all():
    raise RuntimeError("Perimeter identity failed.")
if int(result["pixel_count"].sum()) != 73163:
    raise RuntimeError("Grain pixel area conservation failed.")
result_path = OUT / "ctf_5deg_separated_boundary_exposure.csv"
report_path = OUT / "ctf_5deg_separated_boundary_exposure_report.json"
result.to_csv(result_path, index=False)
report = {
    "grain_count": int(len(result)),
    "valid_pixel_count": int(valid_grid.sum()),
    "grid_dimensions": [int(height), int(width)],
    "map_edge_grains": int(
        (result["map_edge_exposure_sides"] > 0).sum()
    ),
    "unindexed_exposure_grains": int(
        (result["unindexed_neighbour_exposure_sides"] > 0).sum()
    ),
    "grains_with_internal_boundaries": int(
        (result["internal_grain_boundary_sides"] > 0).sum()
    ),
    "total_map_edge_sides": int(
        result["map_edge_exposure_sides"].sum()
    ),
    "total_unindexed_neighbour_sides": int(
        result["unindexed_neighbour_exposure_sides"].sum()
    ),
    "total_internal_grain_boundary_sides": int(
        result["internal_grain_boundary_sides"].sum()
    ),
    "mean_map_edge_exposure_fraction": float(
        result["map_edge_exposure_fraction"].mean()
    ),
    "mean_unindexed_exposure_fraction": float(
        result["unindexed_exposure_fraction"].mean()
    ),
    "mean_internal_boundary_fraction": float(
        result["internal_boundary_fraction"].mean()
    ),
    "perimeter_identity_check": bool(identity.all()),
    "pixel_area_conservation_check": bool(
        int(result["pixel_count"].sum()) == int(valid_grid.sum())
    ),
    "interpretation": [
        "Map-edge exposure and unindexed-neighbour exposure are separate.",
        "Fractions use four-neighbour pixel-side perimeter accounting.",
        "Internal boundaries are counted once for each adjacent grain side.",
        "Measurements remain provisional 2D grid features.",
        "CTF coordinate units remain physically unconfirmed."
    ]
}
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("=" * 72)
print("SEPARATED BOUNDARY EXPOSURE AUDIT")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nExposure distribution:")
print(
    result[
        [
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction",
            "internal_boundary_fraction"
        ]
    ].describe().to_string()
)
print("\nHighest map-edge exposure:")
print(
    result.nlargest(10, "map_edge_exposure_fraction")[
        [
            "grain_id",
            "pixel_count",
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction"
        ]
    ].to_string(index=False)
)
print("\nHighest unindexed-neighbour exposure:")
print(
    result.nlargest(10, "unindexed_exposure_fraction")[
        [
            "grain_id",
            "pixel_count",
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction"
        ]
    ].to_string(index=False)
)
print("\nSaved:")
print(result_path)
print(report_path)
