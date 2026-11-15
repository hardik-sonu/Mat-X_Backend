import json
import numpy as np
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
features = pd.read_csv(root / "ctf_5deg_grain_level_features.csv")
labels = pd.read_csv(root / "ctf_grain_pixels_5deg.csv")
ctf_path = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\EBSD.ctf"
)
labels = labels.sort_values(["Y", "X"]).reset_index(drop=True)
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
nx = data["X"].nunique()
ny = data["Y"].nunique()
grain = labels["grain_id"].to_numpy(dtype=int).reshape(ny, nx)
phase = data["Phase"].to_numpy().reshape(ny, nx)
error = data["Error"].to_numpy().reshape(ny, nx)
valid = (phase == 1) & (error == 0)
if not np.array_equal(valid, grain > 0):
    raise RuntimeError("Segmentation and CTF valid masks differ.")
dx = float(np.median(np.diff(np.sort(data["X"].unique()))))
dy = float(np.median(np.diff(np.sort(data["Y"].unique()))))
max_id = int(grain.max())
map_edge = np.zeros(max_id + 1, dtype=float)
unindexed_edge = np.zeros(max_id + 1, dtype=float)
# Each direction describes one side of each pixel.
directions = [
    (-1, 0, dy, "top"),
    (1, 0, dy, "bottom"),
    (0, -1, dx, "left"),
    (0, 1, dx, "right")
]
for dr, dc, edge_length, name in directions:
    current = grain
    adjacent = np.zeros_like(grain)
    r_start = max(0, -dr)
    r_end = min(ny, ny - dr)
    c_start = max(0, -dc)
    c_end = min(nx, nx - dc)
    adjacent[
        r_start:r_end,
        c_start:c_end
    ] = grain[
        r_start + dr:r_end + dr,
        c_start + dc:c_end + dc
    ]
    rr, cc = np.indices((ny, nx))
    at_map_edge = (
        (rr + dr < 0)
        | (rr + dr >= ny)
        | (cc + dc < 0)
        | (cc + dc >= nx)
    )
    exposed = (current > 0) & (adjacent == 0)
    map_mask = exposed & at_map_edge
    unindexed_mask = exposed & ~at_map_edge
    map_ids = current[map_mask]
    unindexed_ids = current[unindexed_mask]
    map_edge += np.bincount(
        map_ids,
        weights=np.full(len(map_ids), edge_length),
        minlength=max_id + 1
    )[:max_id + 1]
    unindexed_edge += np.bincount(
        unindexed_ids,
        weights=np.full(len(unindexed_ids), edge_length),
        minlength=max_id + 1
    )[:max_id + 1]
# ---------------------------------------------------------
# Attach separate exposure measurements
# ---------------------------------------------------------
result = features.copy()
ids = result["grain_id"].to_numpy(dtype=int)
result["map_edge_length"] = map_edge[ids]
result["unindexed_neighbour_edge_length"] = unindexed_edge[ids]
perimeter = result["perimeter_ctf_coordinate_units"].to_numpy()
result["map_edge_exposure_fraction"] = np.divide(
    result["map_edge_length"],
    perimeter,
    out=np.zeros(len(result), dtype=float),
    where=perimeter > 0
)
result["unindexed_exposure_fraction"] = np.divide(
    result["unindexed_neighbour_edge_length"],
    perimeter,
    out=np.zeros(len(result), dtype=float),
    where=perimeter > 0
)
result["small_grain_flag"] = result["pixel_count"] <= 4
result["map_edge_truncated_flag"] = result["map_edge_length"] > 0
result["substantial_map_edge_exposure_flag"] = (
    result["map_edge_exposure_fraction"] > 0.25
)
result["substantial_unindexed_exposure_flag"] = (
    result["unindexed_exposure_fraction"] > 0.25
)
result["geometry_quality"] = np.select(
    [
        result["small_grain_flag"],
        result["substantial_map_edge_exposure_flag"],
        result["substantial_unindexed_exposure_flag"]
    ],
    [
        "LOW_PIXEL_COUNT",
        "MAP_EDGE_TRUNCATED",
        "UNINDEXED_NEIGHBOUR_EXPOSURE"
    ],
    default="NO_FLAG_FROM_CURRENT_RULES"
)
# Multiple flags may coexist, so preserve independent columns.
result["any_geometry_quality_flag"] = (
    result["small_grain_flag"]
    | result["substantial_map_edge_exposure_flag"]
    | result["substantial_unindexed_exposure_flag"]
)
if not np.allclose(
    result["map_edge_length"]
    + result["unindexed_neighbour_edge_length"],
    result["unindexed_or_map_edge_length_ctf_units"]
):
    raise RuntimeError("External perimeter decomposition failed.")
output = root / "ctf_5deg_grain_geometry_quality_flags.csv"
result.to_csv(output, index=False)
summary = {
    "grain_count": int(len(result)),
    "small_grains_at_most_4_pixels": int(result["small_grain_flag"].sum()),
    "grains_touching_map_edge": int(result["map_edge_truncated_flag"].sum()),
    "grains_with_map_edge_exposure_over_25_percent": int(
        result["substantial_map_edge_exposure_flag"].sum()
    ),
    "grains_with_unindexed_exposure_over_25_percent": int(
        result["substantial_unindexed_exposure_flag"].sum()
    ),
    "grains_with_any_quality_flag": int(
        result["any_geometry_quality_flag"].sum()
    ),
    "map_edge_length_total": float(result["map_edge_length"].sum()),
    "unindexed_neighbour_length_total": float(
        result["unindexed_neighbour_edge_length"].sum()
    ),
    "external_perimeter_total": float(
        result["unindexed_or_map_edge_length_ctf_units"].sum()
    ),
    "decomposition_passed": True,
    "limitations": [
        "Small-grain flag is a screening rule, not proof of segmentation error.",
        "Map-edge exposure indicates truncated observation geometry.",
        "Unindexed exposure does not prove a physical grain boundary.",
        "All grain labels remain provisional."
    ]
}
report = root / "ctf_5deg_grain_geometry_quality_flags.json"
report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("=" * 70)
print("SEPARATE MAP-EDGE / UNINDEXED EXPOSURE AUDIT")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nHighest map-edge exposure:")
print(
    result.nlargest(15, "map_edge_exposure_fraction")[
        ["grain_id", "pixel_count", "map_edge_length",
         "map_edge_exposure_fraction",
         "unindexed_exposure_fraction"]
    ].to_string(index=False)
)
print("\nHighest unindexed exposure:")
print(
    result.nlargest(15, "unindexed_exposure_fraction")[
        ["grain_id", "pixel_count", "map_edge_length",
         "map_edge_exposure_fraction",
         "unindexed_exposure_fraction"]
    ].to_string(index=False)
)
print("\nSaved:", output)
print("Saved:", report)
