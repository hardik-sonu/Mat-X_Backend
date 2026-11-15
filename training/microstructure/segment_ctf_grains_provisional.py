from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.ndimage import label
from itertools import permutations, product
import json
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
ctf = next(root.rglob("*.ctf"))
lines = ctf.read_text(encoding="utf-8", errors="replace").splitlines()
header = next(i for i, x in enumerate(lines)
              if x.strip().startswith("Phase") and "Euler1" in x)
columns = lines[header].split()
records = []
for line in lines[header + 1:]:
    p = line.split()
    if len(p) != len(columns):
        continue
    try:
        records.append([float(x) for x in p])
    except ValueError:
        continue
df = pd.DataFrame(records, columns=columns)
phase = df["Phase"].to_numpy().astype(int)
error = df["Error"].to_numpy().astype(int)
euler = df[["Euler1", "Euler2", "Euler3"]].to_numpy()
nx = int(round(df["X"].nunique()))
ny = int(round(df["Y"].nunique()))
if len(df) != nx * ny:
    raise RuntimeError("Measurement grid is not complete.")
# Verify row-major spatial ordering before reshaping.
xs = np.sort(df["X"].unique())
ys = np.sort(df["Y"].unique())
grid_order = (
    np.allclose(df["X"].to_numpy().reshape(ny, nx)[0], xs)
    and np.allclose(df["Y"].to_numpy().reshape(ny, nx)[:, 0], ys)
)
if not grid_order:
    raise RuntimeError("CTF rows are not in expected row-major grid order.")
valid = ((error == 0) & (phase == 1)).reshape(ny, nx)
# Bunge ZXZ is used provisionally; convention requires confirmation.
angles = np.deg2rad(euler)
orientations = Rotation.from_euler("ZXZ", angles).as_matrix()
orientations = orientations.reshape(ny, nx, 3, 3)
# Generate 24 proper cubic symmetry operators.
symmetry = []
for perm in permutations(range(3)):
    P = np.eye(3)[:, perm]
    for signs in product([-1, 1], repeat=3):
        S = P @ np.diag(signs)
        if round(np.linalg.det(S)) == 1:
            symmetry.append(S)
symmetry = np.asarray(symmetry)
# Four-neighbor connected-component segmentation.
structure = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=int)
# Precompute pairwise neighbor misorientation angles.
edge_angles = []
for dy, dx in [(0, 1), (1, 0)]:
    a = orientations[:ny-dy or None, :nx-dx or None]
    b = orientations[dy:, dx:]
    pair_valid = valid[:ny-dy or None, :nx-dx or None] & valid[dy:, dx:]
    # Minimum disorientation over cubic symmetry equivalents.
    min_angle = np.full(pair_valid.shape, np.inf)
    for S in symmetry:
        relative = np.einsum("...ji,jk,...kl->...il", a, S, b)
        trace = np.trace(relative, axis1=-2, axis2=-1)
        cosine = np.clip((trace - 1) / 2, -1, 1)
        angle = np.degrees(np.arccos(cosine))
        min_angle = np.minimum(min_angle, angle)
    edge_angles.append((dy, dx, pair_valid, min_angle))
results = []
for threshold in [2, 5, 10, 15]:
    links = np.zeros((ny, nx), dtype=bool)
    # Build a graph of accepted neighboring pairs.
    parent = np.arange(ny * nx, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for dy, dx, pair_valid, angles_grid in edge_angles:
        accepted = pair_valid & (angles_grid <= threshold)
        yy, xx = np.where(accepted)
        for y, x in zip(yy, xx):
            a = y * nx + x
            b = (y + dy) * nx + (x + dx)
            union(a, b)
    valid_flat = valid.ravel()
    roots = {}
    labels = np.zeros(ny * nx, dtype=np.int32)
    next_id = 1
    for idx in np.flatnonzero(valid_flat):
        r = find(int(idx))
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        labels[idx] = roots[r]
    label_grid = labels.reshape(ny, nx)
    counts = np.bincount(labels[labels > 0])
    grain_sizes = counts[1:] if len(counts) > 1 else np.array([])
    results.append({
        "threshold_degrees": threshold,
        "grain_count": int(len(grain_sizes)),
        "indexed_pixels": int(valid.sum()),
        "mean_pixels_per_grain": float(grain_sizes.mean()) if len(grain_sizes) else None,
        "median_pixels_per_grain": float(np.median(grain_sizes)) if len(grain_sizes) else None,
        "smallest_grain_pixels": int(grain_sizes.min()) if len(grain_sizes) else None,
        "largest_grain_pixels": int(grain_sizes.max()) if len(grain_sizes) else None,
        "grains_under_5_pixels": int((grain_sizes < 5).sum()) if len(grain_sizes) else 0
    })
    pd.DataFrame({
        "pixel_index": np.arange(ny * nx),
        "grain_label": labels
    }).to_csv(
        Path("training_results/microstructure/abaqus_deep_audit")
        / f"ctf_provisional_grain_labels_{threshold}deg.csv",
        index=False
    )
out = Path("training_results/microstructure/abaqus_deep_audit")
out.mkdir(parents=True, exist_ok=True)
report = {
    "source": str(ctf),
    "grid": [nx, ny],
    "indexed_points": int(valid.sum()),
    "euler_convention_used": "Bunge ZXZ, provisional",
    "crystal_symmetry": "cubic, 24 proper rotations",
    "segmentation": "4-neighbor connected components",
    "threshold_sensitivity": results,
    "physical_validation": "NOT_YET_ESTABLISHED"
}
(out / "ctf_provisional_grain_segmentation_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("PROVISIONAL EBSD GRAIN SEGMENTATION")
print(pd.DataFrame(results).to_string(index=False))
print("\nConvention: provisional Bunge ZXZ")
print("Crystal symmetry: cubic")
print("Saved report:", out / "ctf_provisional_grain_segmentation_report.json")
print("Original CTF unchanged.")
