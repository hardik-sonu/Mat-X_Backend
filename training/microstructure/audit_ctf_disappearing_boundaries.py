import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations, product
from scipy.spatial.transform import Rotation
root = Path("training_results/microstructure/abaqus_deep_audit")
ctf = pd.read_csv(root / "ctf_measurements_audited.csv")
ctf = ctf.sort_values(["Y", "X"]).reset_index(drop=True)
nx = ctf["X"].nunique()
ny = ctf["Y"].nunique()
if len(ctf) != nx * ny:
    raise RuntimeError("Incomplete CTF grid.")
valid = (
    (ctf["Phase"].to_numpy().reshape(ny, nx) == 1)
    & (ctf["Error"].to_numpy().reshape(ny, nx) == 0)
)
euler = ctf[["Euler1", "Euler2", "Euler3"]].to_numpy()
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
def calculate_angles(a, b):
    minimum = np.full(a.shape[:-2], np.inf)
    for S in symmetry:
        relative = np.einsum(
            "...ji,jk,...kl->...il", a, S, b
        )
        trace = np.trace(relative, axis1=-2, axis2=-1)
        cosine = np.clip((trace - 1) / 2, -1, 1)
        angle = np.degrees(np.arccos(cosine))
        minimum = np.minimum(minimum, angle)
    return minimum
records = []
for lower, upper in [(2, 5), (5, 10), (10, 15)]:
    low = pd.read_csv(root / f"ctf_grain_pixels_{lower}deg.csv")
    high = pd.read_csv(root / f"ctf_grain_pixels_{upper}deg.csv")
    low = low.sort_values(["Y", "X"]).reset_index(drop=True)
    high = high.sort_values(["Y", "X"]).reset_index(drop=True)
    low_labels = low["grain_id"].to_numpy().reshape(ny, nx)
    high_labels = high["grain_id"].to_numpy().reshape(ny, nx)
    for dy, dx in [(0, 1), (1, 0)]:
        a = orientations[:ny-dy or None, :nx-dx or None]
        b = orientations[dy:, dx:]
        lv = low_labels[:ny-dy or None, :nx-dx or None]
        rv = low_labels[dy:, dx:]
        lu = high_labels[:ny-dy or None, :nx-dx or None]
        ru = high_labels[dy:, dx:]
        pair_valid = (
            valid[:ny-dy or None, :nx-dx or None]
            & valid[dy:, dx:]
        )
        # Boundary links that disappear at the higher threshold.
        newly_merged = (
            pair_valid
            & (lv > 0)
            & (rv > 0)
            & (lv != rv)
            & (lu == ru)
        )
        if not newly_merged.any():
            continue
        angles = calculate_angles(
            a[newly_merged],
            b[newly_merged]
        )
        left_ids = lv[newly_merged]
        right_ids = rv[newly_merged]
        upper_ids = lu[newly_merged]
        for upper_id in np.unique(upper_ids):
            select = upper_ids == upper_id
            values = angles[select]
            lower_ids = sorted(set(
                left_ids[select].tolist()
                + right_ids[select].tolist()
            ))
            records.append({
                "lower_threshold": lower,
                "upper_threshold": upper,
                "upper_component": int(upper_id),
                "lower_component_ids": ",".join(
                    str(int(v)) for v in lower_ids
                ),
                "boundary_pixel_pairs": int(len(values)),
                "minimum_angle": float(values.min()),
                "median_angle": float(np.median(values)),
                "mean_angle": float(values.mean()),
                "maximum_angle": float(values.max()),
                "pairs_below_2deg": int((values < 2).sum()),
                "pairs_2_to_5deg": int(
                    ((values >= 2) & (values < 5)).sum()
                ),
                "pairs_5_to_10deg": int(
                    ((values >= 5) & (values < 10)).sum()
                ),
                "pairs_10_to_15deg": int(
                    ((values >= 10) & (values < 15)).sum()
                ),
                "pairs_15deg_or_more": int((values >= 15).sum())
            })
result = pd.DataFrame(records)
output = root / "ctf_disappearing_boundary_angle_audit.csv"
result.to_csv(output, index=False)
print("=" * 75)
print("DISAPPEARING GRAIN-BOUNDARY MISORIENTATION AUDIT")
print("=" * 75)
if result.empty:
    raise RuntimeError("No disappearing boundaries were identified.")
for lower, upper in [(2, 5), (5, 10), (10, 15)]:
    subset = result[
        (result["lower_threshold"] == lower)
        & (result["upper_threshold"] == upper)
    ]
    print(f"\n{lower}° TO {upper}°")
    print("Merged boundaries:", len(subset))
    if len(subset):
        print(
            subset.sort_values(
                "boundary_pixel_pairs",
                ascending=False
            ).head(15).to_string(index=False)
        )
print("\nSaved:", output)
print("Original CTF and segmentation files unchanged.")
