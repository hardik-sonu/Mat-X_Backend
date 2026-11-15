from pathlib import Path
import json
from collections import Counter
import numpy as np
import pandas as pd
OUT = Path("training_results/microstructure/abaqus_deep_audit")
pixels = pd.read_csv(OUT / "ctf_grain_pixels_5deg.csv")
features = pd.read_csv(OUT / "ctf_5deg_grain_level_features_v3.csv")
pairs = pd.read_csv(OUT / "ctf_5deg_grain_neighbour_pairs.csv")
required = {"X", "Y", "Phase", "Error", "grain_id"}
if not required.issubset(pixels.columns):
    raise RuntimeError(f"Missing pixel columns: {required - set(pixels.columns)}")
pixels = pixels.sort_values(["Y", "X"]).reset_index(drop=True)
xs = np.sort(pixels["X"].unique())
ys = np.sort(pixels["Y"].unique())
height, width = len(ys), len(xs)
if (height, width) != (272, 272):
    raise RuntimeError(f"Unexpected grid dimensions: {(height, width)}")
if len(pixels) != height * width:
    raise RuntimeError("Pixel grid is incomplete.")
labels = pixels["grain_id"].to_numpy(dtype=int).reshape(height, width)
valid = labels > 0
if int(valid.sum()) != 73163:
    raise RuntimeError("Unexpected indexed pixel population.")
# Reconstruct all horizontal and vertical grain interfaces.
horizontal_a = labels[:, :-1]
horizontal_b = labels[:, 1:]
vertical_a = labels[:-1, :]
vertical_b = labels[1:, :]
horizontal_mask = (
    (horizontal_a > 0)
    & (horizontal_b > 0)
    & (horizontal_a != horizontal_b)
)
vertical_mask = (
    (vertical_a > 0)
    & (vertical_b > 0)
    & (vertical_a != vertical_b)
)
contact_counts = Counter()
grain_side_counts = Counter()
def record_contacts(a, b, mask):
    left = a[mask].astype(int)
    right = b[mask].astype(int)
    for g1, g2 in zip(left, right):
        pair = tuple(sorted((int(g1), int(g2))))
        contact_counts[pair] += 1
        grain_side_counts[pair[0]] += 1
        grain_side_counts[pair[1]] += 1
record_contacts(horizontal_a, horizontal_b, horizontal_mask)
record_contacts(vertical_a, vertical_b, vertical_mask)
reconstructed_segments = int(horizontal_mask.sum() + vertical_mask.sum())
reconstructed_pairs = len(contact_counts)
reconstructed_degrees = Counter()
for g1, g2 in contact_counts:
    reconstructed_degrees[g1] += 1
    reconstructed_degrees[g2] += 1
pair_rows = pairs.copy()
pair_rows["pair_key"] = [
    tuple(sorted((int(a), int(b))))
    for a, b in zip(pair_rows["grain_1"], pair_rows["grain_2"])
]
stored_pair_counts = {
    row.pair_key: int(row.boundary_pixel_pairs)
    for row in pair_rows.itertuples()
}
stored_pairs = set(stored_pair_counts)
rebuilt_pairs = set(contact_counts)
pair_sets_match = stored_pairs == rebuilt_pairs
pair_contact_counts_match = (
    pair_sets_match
    and all(
        stored_pair_counts[pair] == contact_counts[pair]
        for pair in rebuilt_pairs
    )
)
feature_degrees = dict(zip(
    features["grain_id"].astype(int),
    features["neighbour_count"].astype(int)
))
feature_contacts = dict(zip(
    features["grain_id"].astype(int),
    features["boundary_segment_count"].astype(int)
))
all_grains = set(features["grain_id"].astype(int))
degree_match = all(
    feature_degrees.get(g, 0) == reconstructed_degrees.get(g, 0)
    for g in all_grains
)
contact_match = all(
    feature_contacts.get(g, 0) == grain_side_counts.get(g, 0)
    for g in all_grains
)
checks = {
    "grid_dimensions_272_by_272": (height, width) == (272, 272),
    "valid_pixel_count_73163": int(valid.sum()) == 73163,
    "reconstructed_segments_14248": reconstructed_segments == 14248,
    "reconstructed_pairs_1752": reconstructed_pairs == 1752,
    "pair_identity_matches_stored_network": pair_sets_match,
    "pair_contact_counts_match": pair_contact_counts_match,
    "per_grain_neighbour_degrees_match": degree_match,
    "per_grain_boundary_contacts_match": contact_match,
    "all_652_grains_accounted_for": len(all_grains) == 652
}
report = {
    "grid_dimensions": [height, width],
    "valid_pixels": int(valid.sum()),
    "reconstructed_boundary_segments": reconstructed_segments,
    "reconstructed_unique_pairs": reconstructed_pairs,
    "reconstructed_grain_side_contacts": int(sum(grain_side_counts.values())),
    "stored_pair_count": len(stored_pairs),
    "missing_pairs_in_stored_network": len(rebuilt_pairs - stored_pairs),
    "extra_pairs_in_stored_network": len(stored_pairs - rebuilt_pairs),
    "maximum_pair_contact_count_difference": (
        max(
            abs(stored_pair_counts[p] - contact_counts[p])
            for p in rebuilt_pairs
        )
        if pair_sets_match and rebuilt_pairs else None
    ),
    "integrity_checks": checks,
    "all_checks_pass": all(checks.values()),
    "method": "Independent four-neighbour reconstruction from grain pixel labels",
    "limitations": [
        "Topology validation does not validate Euler-angle convention.",
        "Topology validation does not establish physical grain boundaries.",
        "Segmentation remains provisional at 5 degrees."
    ]
}
report_path = OUT / "ctf_5deg_independent_topology_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("=" * 72)
print("INDEPENDENT PIXEL-GRID TOPOLOGY AUDIT")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nIndividual checks:")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print("\nSaved:")
print(report_path.resolve())
if not all(checks.values()):
    raise RuntimeError("Independent topology audit requires review.")
