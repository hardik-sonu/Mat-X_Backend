import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations, product
from scipy.spatial.transform import Rotation
root = Path("training_results/microstructure/abaqus_deep_audit")
ctf_path = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\EBSD.ctf"
)
# ---------------------------------------------------------
# 1. Read and validate CTF header
# ---------------------------------------------------------
with ctf_path.open("r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
header_index = None
for i, line in enumerate(lines):
    tokens = line.split()
    if (
        len(tokens) >= 8
        and tokens[0] == "Phase"
        and tokens[1] == "X"
        and "Euler1" in tokens
        and "Euler2" in tokens
        and "Euler3" in tokens
    ):
        header_index = i
        break
if header_index is None:
    raise RuntimeError("Could not identify CTF measurement header.")
columns = lines[header_index].split()
data = pd.read_csv(
    ctf_path,
    sep=r"\s+",
    skiprows=header_index + 1,
    names=columns,
    header=None,
    engine="python"
)
required = {
    "Phase", "X", "Y", "Error",
    "Euler1", "Euler2", "Euler3"
}
if not required.issubset(data.columns):
    raise RuntimeError(f"Missing CTF columns: {required - set(data.columns)}")
data = data.sort_values(["Y", "X"]).reset_index(drop=True)
nx = data["X"].nunique()
ny = data["Y"].nunique()
if len(data) != nx * ny:
    raise RuntimeError("CTF grid dimensions do not match measurement count.")
print("CTF header:", columns)
print("Grid:", nx, "x", ny)
print("Measurements:", len(data))
# ---------------------------------------------------------
# 2. Load 5-degree grain segmentation
# ---------------------------------------------------------
labels = pd.read_csv(root / "ctf_grain_pixels_5deg.csv")
labels = labels.sort_values(["Y", "X"]).reset_index(drop=True)
if len(labels) != len(data):
    raise RuntimeError("CTF and grain-label row counts differ.")
if not np.allclose(
    labels[["X", "Y"]].to_numpy(),
    data[["X", "Y"]].to_numpy()
):
    raise RuntimeError("CTF and segmentation coordinates do not align.")
grain = labels["grain_id"].to_numpy().reshape(ny, nx)
phase = data["Phase"].to_numpy().reshape(ny, nx)
error = data["Error"].to_numpy().reshape(ny, nx)
valid = (phase == 1) & (error == 0)
if int(valid.sum()) != 73163:
    raise RuntimeError("Unexpected valid-pixel count.")
if np.any(grain[valid] <= 0):
    raise RuntimeError("Valid pixels have missing grain labels.")
# ---------------------------------------------------------
# 3. Calculate orientation matrices
# ---------------------------------------------------------
euler = data[
    ["Euler1", "Euler2", "Euler3"]
].to_numpy(dtype=float)
orientation = Rotation.from_euler(
    "ZXZ",
    np.deg2rad(euler)
).as_matrix().reshape(ny, nx, 3, 3)
# ---------------------------------------------------------
# 4. Generate proper cubic symmetry operators
# ---------------------------------------------------------
symmetry = []
for perm in permutations(range(3)):
    P = np.eye(3)[:, perm]
    for signs in product([-1, 1], repeat=3):
        S = P @ np.diag(signs)
        if round(np.linalg.det(S)) == 1:
            symmetry.append(S)
symmetry = np.asarray(symmetry)
if len(symmetry) != 24:
    raise RuntimeError("Expected 24 cubic proper symmetry operators.")
# ---------------------------------------------------------
# 5. Vectorized disorientation calculation
# ---------------------------------------------------------
def calculate_disorientation(A, B):
    minimum = np.full(len(A), np.inf)
    for S in symmetry:
        relative = np.einsum(
            "nij,jk,nkl->nil",
            np.transpose(A, (0, 2, 1)),
            S,
            B
        )
        trace = np.trace(relative, axis1=1, axis2=2)
        cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        minimum = np.minimum(minimum, angles)
    return minimum
# ---------------------------------------------------------
# 6. Extract horizontal and vertical grain boundaries
# ---------------------------------------------------------
boundary_parts = []
for direction, dy, dx in [
    ("horizontal", 0, 1),
    ("vertical", 1, 0)
]:
    if direction == "horizontal":
        valid_a = valid[:, :-1]
        valid_b = valid[:, 1:]
        grain_a = grain[:, :-1]
        grain_b = grain[:, 1:]
        ori_a = orientation[:, :-1]
        ori_b = orientation[:, 1:]
        x_a = data["X"].to_numpy().reshape(ny, nx)[:, :-1]
        y_a = data["Y"].to_numpy().reshape(ny, nx)[:, :-1]
    else:
        valid_a = valid[:-1, :]
        valid_b = valid[1:, :]
        grain_a = grain[:-1, :]
        grain_b = grain[1:, :]
        ori_a = orientation[:-1, :]
        ori_b = orientation[1:, :]
        x_a = data["X"].to_numpy().reshape(ny, nx)[:-1, :]
        y_a = data["Y"].to_numpy().reshape(ny, nx)[:-1, :]
    boundary_mask = (
        valid_a
        & valid_b
        & (grain_a > 0)
        & (grain_b > 0)
        & (grain_a != grain_b)
    )
    if not boundary_mask.any():
        continue
    A = ori_a[boundary_mask]
    B = ori_b[boundary_mask]
    angles = calculate_disorientation(A, B)
    ids_a = grain_a[boundary_mask].astype(int)
    ids_b = grain_b[boundary_mask].astype(int)
    boundary_parts.append(pd.DataFrame({
        "grain_1": np.minimum(ids_a, ids_b),
        "grain_2": np.maximum(ids_a, ids_b),
        "misorientation_degrees": angles,
        "direction": direction,
        "pixel_x": x_a[boundary_mask],
        "pixel_y": y_a[boundary_mask]
    }))
    print(
        direction,
        "boundary segments:",
        len(angles)
    )
if not boundary_parts:
    raise RuntimeError("No grain boundaries detected.")
boundaries = pd.concat(boundary_parts, ignore_index=True)
# ---------------------------------------------------------
# 7. Classify boundary misorientation
# ---------------------------------------------------------
boundaries["boundary_class"] = np.select(
    [
        boundaries["misorientation_degrees"] < 2,
        boundaries["misorientation_degrees"] < 15
    ],
    [
        "below_2_degrees",
        "low_angle_2_to_15"
    ],
    default="high_angle_15_or_more"
)
# ---------------------------------------------------------
# 8. Aggregate unique grain-neighbour pairs
# ---------------------------------------------------------
pairs = (
    boundaries.groupby(
        ["grain_1", "grain_2"],
        as_index=False
    )
    .agg(
        boundary_pixel_pairs=("misorientation_degrees", "size"),
        mean_misorientation=("misorientation_degrees", "mean"),
        median_misorientation=("misorientation_degrees", "median"),
        min_misorientation=("misorientation_degrees", "min"),
        max_misorientation=("misorientation_degrees", "max")
    )
)
# ---------------------------------------------------------
# 9. Grain-level neighbour statistics
# ---------------------------------------------------------
neighbours = {int(g): set() for g in np.unique(grain[valid])}
contacts = {int(g): 0 for g in np.unique(grain[valid])}
for row in pairs.itertuples(index=False):
    a = int(row.grain_1)
    b = int(row.grain_2)
    neighbours[a].add(b)
    neighbours[b].add(a)
    contacts[a] += int(row.boundary_pixel_pairs)
    contacts[b] += int(row.boundary_pixel_pairs)
grain_table = pd.DataFrame([
    {
        "grain_id": gid,
        "neighbour_count": len(neighbours[gid]),
        "boundary_pixel_contacts": contacts[gid]
    }
    for gid in sorted(neighbours)
])
# ---------------------------------------------------------
# 10. Triple-junction candidate count
# ---------------------------------------------------------
# Count grid vertices where three or more distinct valid
# grain labels meet. This is a pixel-grid estimate, not
# a subpixel metallographic triple-junction measurement.
junction_counts = {}
for y in range(ny - 1):
    for x in range(nx - 1):
        block_valid = valid[y:y+2, x:x+2]
        if not block_valid.all():
            continue
        block = grain[y:y+2, x:x+2]
        unique = np.unique(block)
        if len(unique) >= 3:
            junction_counts[len(unique)] = (
                junction_counts.get(len(unique), 0) + 1
            )
# ---------------------------------------------------------
# 11. Save outputs
# ---------------------------------------------------------
boundaries.to_csv(
    root / "ctf_5deg_grain_boundary_segments.csv",
    index=False
)
pairs.to_csv(
    root / "ctf_5deg_grain_neighbour_pairs.csv",
    index=False
)
grain_table.to_csv(
    root / "ctf_5deg_grain_neighbour_statistics.csv",
    index=False
)
class_counts = (
    boundaries["boundary_class"]
    .value_counts()
    .to_dict()
)
summary = {
    "segmentation_threshold_degrees": 5,
    "orientation_convention": "Provisional ZXZ",
    "symmetry_group": "Cubic proper rotations, 24 operators",
    "grain_count": int(len(grain_table)),
    "valid_pixels": int(valid.sum()),
    "boundary_pixel_segments": int(len(boundaries)),
    "unique_neighbour_pairs": int(len(pairs)),
    "boundary_class_counts": {
        str(k): int(v) for k, v in class_counts.items()
    },
    "misorientation_statistics_degrees": {
        "mean": float(boundaries["misorientation_degrees"].mean()),
        "median": float(boundaries["misorientation_degrees"].median()),
        "p90": float(boundaries["misorientation_degrees"].quantile(.90)),
        "p95": float(boundaries["misorientation_degrees"].quantile(.95)),
        "maximum": float(boundaries["misorientation_degrees"].max())
    },
    "grain_neighbour_statistics": {
        "mean": float(grain_table["neighbour_count"].mean()),
        "median": float(grain_table["neighbour_count"].median()),
        "minimum": int(grain_table["neighbour_count"].min()),
        "maximum": int(grain_table["neighbour_count"].max())
    },
    "triple_junction_pixel_candidates": {
        str(k): int(v) for k, v in junction_counts.items()
    },
    "limitations": [
        "Euler convention and physical reference frame require source verification.",
        "Grain segmentation is provisional.",
        "Boundary lengths are pixel-grid estimates.",
        "Triple junctions are pixel-grid candidates, not subpixel measurements."
    ]
}
(root / "ctf_5deg_grain_boundary_network_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("\n" + "=" * 70)
print("5-DEGREE GRAIN BOUNDARY NETWORK SUMMARY")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nSaved:")
print(root / "ctf_5deg_grain_boundary_segments.csv")
print(root / "ctf_5deg_grain_neighbour_pairs.csv")
print(root / "ctf_5deg_grain_neighbour_statistics.csv")
print(root / "ctf_5deg_grain_boundary_network_summary.json")
print("\nOriginal CTF and previous segmentation outputs unchanged.")
