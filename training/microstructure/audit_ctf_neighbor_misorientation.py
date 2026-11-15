import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations, product
from scipy.spatial.transform import Rotation
root = Path("training_results/microstructure/abaqus_deep_audit")
source = root / "ctf_measurements_audited.csv"
df = pd.read_csv(source)
required = [
    "X", "Y", "Phase", "Error",
    "Euler1", "Euler2", "Euler3"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(
        f"Missing columns: {missing}. Available: {list(df.columns)}"
    )
df = df.sort_values(["Y", "X"]).reset_index(drop=True)
nx = df["X"].nunique()
ny = df["Y"].nunique()
if len(df) != nx * ny:
    raise RuntimeError("CTF grid is incomplete.")
if df.duplicated(["X", "Y"]).any():
    raise RuntimeError("Duplicate grid coordinates detected.")
phase = df["Phase"].to_numpy().reshape(ny, nx)
error = df["Error"].to_numpy().reshape(ny, nx)
valid = (phase == 1) & (error == 0)
euler = df[["Euler1", "Euler2", "Euler3"]].to_numpy()
orientations = Rotation.from_euler(
    "ZXZ", np.deg2rad(euler)
).as_matrix().reshape(ny, nx, 3, 3)
symmetry = []
for perm in permutations(range(3)):
    P = np.eye(3)[:, perm]
    for signs in product([-1, 1], repeat=3):
        S = P @ np.diag(signs)
        if round(np.linalg.det(S)) == 1:
            symmetry.append(S)
symmetry = np.asarray(symmetry)
if len(symmetry) != 24:
    raise RuntimeError("Expected 24 proper cubic symmetry operators.")
angle_arrays = []
for dy, dx in [(0, 1), (1, 0)]:
    a = orientations[:ny-dy or None, :nx-dx or None]
    b = orientations[dy:, dx:]
    pair_valid = (
        valid[:ny-dy or None, :nx-dx or None]
        & valid[dy:, dx:]
    )
    min_angle = np.full(pair_valid.shape, np.nan)
    for S in symmetry:
        relative = np.einsum(
            "...ji,jk,...kl->...il", a, S, b
        )
        trace = np.trace(relative, axis1=-2, axis2=-1)
        cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        min_angle = np.fmin(min_angle, angle)
    angle_arrays.append(min_angle[pair_valid])
angles = np.concatenate(angle_arrays)
if not np.isfinite(angles).all():
    raise RuntimeError("Non-finite neighbour misorientation detected.")
thresholds = [2, 5, 10, 15]
summary = {
    "grid": [int(nx), int(ny)],
    "valid_pixels": int(valid.sum()),
    "valid_neighbour_pairs": int(len(angles)),
    "symmetry": "24 proper cubic rotations",
    "euler_convention": "Bunge ZXZ, subject to CTF export verification",
    "angle_statistics_degrees": {
        "minimum": float(angles.min()),
        "maximum": float(angles.max()),
        "mean": float(angles.mean()),
        "median": float(np.median(angles)),
        "p75": float(np.percentile(angles, 75)),
        "p90": float(np.percentile(angles, 90)),
        "p95": float(np.percentile(angles, 95)),
        "p99": float(np.percentile(angles, 99)),
    },
    "threshold_analysis": []
}
for threshold in thresholds:
    accepted = int(np.count_nonzero(angles <= threshold))
    rejected = int(np.count_nonzero(angles > threshold))
    summary["threshold_analysis"].append({
        "threshold_degrees": threshold,
        "within_threshold_pairs": accepted,
        "above_threshold_pairs": rejected,
        "within_percent": 100.0 * accepted / len(angles),
        "above_percent": 100.0 * rejected / len(angles)
    })
hist_edges = np.arange(0, 62, 1)
hist_counts, edges = np.histogram(angles, bins=hist_edges)
histogram = pd.DataFrame({
    "angle_bin_start_degrees": edges[:-1],
    "angle_bin_end_degrees": edges[1:],
    "neighbor_pair_count": hist_counts
})
histogram.to_csv(
    root / "ctf_neighbor_misorientation_histogram.csv",
    index=False
)
np.save(
    root / "ctf_neighbor_misorientation_angles.npy",
    angles
)
(root / "ctf_neighbor_misorientation_audit.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 70)
print("CTF NEIGHBOUR MISORIENTATION AUDIT")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nHistogram saved:")
print(root / "ctf_neighbor_misorientation_histogram.csv")
print("\nRaw angles saved:")
print(root / "ctf_neighbor_misorientation_angles.npy")
print("\nOriginal CTF unchanged.")
