from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import permutations, product
import json
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
source = next(root.rglob("*.ctf"))
lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
header = next(i for i, line in enumerate(lines)
              if line.strip().startswith("Phase") and "Euler1" in line)
columns = lines[header].split()
rows = []
for line in lines[header + 1:]:
    parts = line.split()
    if len(parts) != len(columns):
        continue
    try:
        rows.append([float(v) for v in parts])
    except ValueError:
        continue
df = pd.DataFrame(rows, columns=columns)
# Explicitly sort the measurements into their spatial grid.
df = df.sort_values(["Y", "X"]).reset_index(drop=True)
xs = np.sort(df["X"].unique())
ys = np.sort(df["Y"].unique())
nx, ny = len(xs), len(ys)
if len(df) != nx * ny:
    raise RuntimeError("Incomplete spatial grid.")
if df.duplicated(["X", "Y"]).any():
    raise RuntimeError("Duplicate spatial coordinates found.")
valid = ((df["Phase"] == 1) & (df["Error"] == 0)).to_numpy()
if int(valid.sum()) != 73163:
    raise RuntimeError("Expected 73,163 valid FCC orientation points.")
valid_grid = valid.reshape(ny, nx)
# CTF coordinate spacing; physical units are not established.
x_step = float(np.median(np.diff(xs)))
y_step = float(np.median(np.diff(ys)))
pixel_area = x_step * y_step
# Provisional Bunge ZXZ convention.
euler = df[["Euler1", "Euler2", "Euler3"]].to_numpy()
rotations = Rotation.from_euler("ZXZ", np.deg2rad(euler)).as_matrix()
rotations = rotations.reshape(ny, nx, 3, 3)
# 24 proper cubic symmetry operators.
symmetry = []
for perm in permutations(range(3)):
    P = np.eye(3)[:, perm]
    for signs in product([-1, 1], repeat=3):
        S = P @ np.diag(signs)
        if round(np.linalg.det(S)) == 1:
            symmetry.append(S)
symmetry = np.asarray(symmetry)
# Calculate neighboring disorientation values once.
edges = []
for dy, dx in [(0, 1), (1, 0)]:
    a = rotations[:ny-dy or None, :nx-dx or None]
    b = rotations[dy:, dx:]
    pair_valid = (
        valid_grid[:ny-dy or None, :nx-dx or None]
        & valid_grid[dy:, dx:]
    )
    minimum = np.full(pair_valid.shape, np.inf)
    for S in symmetry:
        relative = np.einsum("...ji,jk,...kl->...il", a, S, b)
        trace = np.trace(relative, axis1=-2, axis2=-1)
        cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        minimum = np.minimum(minimum, angle)
    edges.append((dy, dx, pair_valid, minimum))
out = Path("training_results/microstructure/abaqus_deep_audit")
out.mkdir(parents=True, exist_ok=True)
all_reports = []
for threshold in [2, 5, 10, 15]:
    parent = np.arange(ny * nx, dtype=np.int64)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for dy, dx, pair_valid, angle_grid in edges:
        yy, xx = np.where(pair_valid & (angle_grid <= threshold))
        for y, x in zip(yy, xx):
            union(y * nx + x, (y + dy) * nx + (x + dx))
    labels = np.zeros(ny * nx, dtype=np.int32)
    root_to_label = {}
    next_label = 1
    for idx in np.flatnonzero(valid):
        r = find(int(idx))
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[idx] = root_to_label[r]
    label_grid = labels.reshape(ny, nx)
    # Pixel-level spatial labels.
    pixel_output = df[["X", "Y", "Phase", "Error"]].copy()
    pixel_output["grain_id"] = labels
    pixel_output.to_csv(
        out / f"ctf_grain_pixels_{threshold}deg.csv",
        index=False
    )
    # Grain-level geometric descriptors.
    grain_records = []
    for grain_id in range(1, next_label):
        gy, gx = np.where(label_grid == grain_id)
        count = len(gx)
        x_values = xs[gx]
        y_values = ys[gy]
        area = count * pixel_area
        eq_diameter = np.sqrt(4.0 * area / np.pi)
        grain_records.append({
            "grain_id": grain_id,
            "pixel_count": count,
            "area_ctf_coordinate_units_squared": area,
            "equivalent_diameter_ctf_coordinate_units": eq_diameter,
            "centroid_x": float(x_values.mean()),
            "centroid_y": float(y_values.mean()),
            "x_min": float(x_values.min()),
            "x_max": float(x_values.max()),
            "y_min": float(y_values.min()),
            "y_max": float(y_values.max())
        })
    grain_df = pd.DataFrame(grain_records)
    grain_df.to_csv(
        out / f"ctf_grain_geometry_{threshold}deg.csv",
        index=False
    )
    all_reports.append({
        "threshold_degrees": threshold,
        "grain_count": int(len(grain_df)),
        "indexed_pixels": int(grain_df["pixel_count"].sum()),
        "mean_pixels_per_grain": float(grain_df["pixel_count"].mean()),
        "median_pixels_per_grain": float(grain_df["pixel_count"].median()),
        "minimum_pixels": int(grain_df["pixel_count"].min()),
        "maximum_pixels": int(grain_df["pixel_count"].max()),
        "grains_under_5_pixels": int((grain_df["pixel_count"] < 5).sum()),
        "median_equivalent_diameter_coordinate_units": float(
            grain_df["equivalent_diameter_ctf_coordinate_units"].median()
        )
    })
report = {
    "source_file": source.name,
    "grid": [nx, ny],
    "valid_orientation_points": int(valid.sum()),
    "validity_rule": "Phase == 1 AND Error == 0",
    "x_step_from_ctf": x_step,
    "y_step_from_ctf": y_step,
    "coordinate_units": "NOT_ESTABLISHED",
    "euler_convention": "PROVISIONAL_BUNGE_ZXZ",
    "crystal_symmetry": "CUBIC_24_PROPER_ROTATIONS",
    "segmentation": "FOUR_NEIGHBOR_CONNECTED_COMPONENTS",
    "threshold_results": all_reports,
    "grain_geometry_status": "PROVISIONAL_NOT_PHYSICALLY_VALIDATED"
}
(out / "ctf_grain_geometry_summary.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("SPATIAL GRAIN RECORD GENERATION COMPLETE")
print(pd.DataFrame(all_reports).to_string(index=False))
print("\nOutput directory:", out)
print("Original CTF unchanged.")
