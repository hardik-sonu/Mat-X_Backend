import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
ROOT = Path("training_results/microstructure/abaqus_deep_audit")
INPUT = ROOT / "ctf_5deg_matx_grain_features_quality_annotated.csv"
K = 8
PERMUTATIONS = 999
SEED = 42
df = pd.read_csv(INPUT)
features = [
    "pixel_count",
    "equivalent_diameter_ctf_coordinate_units",
    "circularity",
    "aspect_ratio_bbox",
    "neighbour_count",
    "mean_boundary_misorientation",
    "high_angle_boundary_fraction",
    "boundary_exposure_fraction",
]
required = ["centroid_x", "centroid_y", "geometry_screen_pass"]
missing = [c for c in required + features if c not in df.columns]
if missing:
    raise RuntimeError(f"Required columns missing: {missing}")
def build_spatial_weights(frame, k):
    coords = frame[["centroid_x", "centroid_y"]].to_numpy(float)
    if len(coords) <= k:
        raise ValueError("Not enough grains for requested KNN graph.")
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    rows = np.repeat(np.arange(len(coords)), k)
    cols = indices[:, 1:].reshape(-1)
    weights = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(coords), len(coords))
    )
    # Symmetric union graph.
    weights = weights.maximum(weights.T)
    weights.setdiag(0)
    weights.eliminate_zeros()
    return weights
def moran_test(values, weights, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 10 or np.std(values) == 0:
        return {
            "n": n,
            "moran_i": None,
            "permutation_p_two_sided": None,
            "status": "INSUFFICIENT_OR_CONSTANT_FEATURE"
        }
    z = values - values.mean()
    denominator = float(z @ z)
    s0 = float(weights.sum())
    observed = (n / s0) * float(z @ (weights @ z)) / denominator
    simulated = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        zp = rng.permutation(z)
        simulated[i] = (
            (n / s0)
            * float(zp @ (weights @ zp))
            / denominator
        )
    p_value = (
        np.count_nonzero(np.abs(simulated) >= abs(observed)) + 1
    ) / (PERMUTATIONS + 1)
    return {
        "n": n,
        "moran_i": float(observed),
        "permutation_p_two_sided": float(p_value),
        "permutation_mean": float(simulated.mean()),
        "permutation_std": float(simulated.std(ddof=1)),
        "status": "COMPUTED"
    }
rng = np.random.default_rng(SEED)
results = []
subsets = {
    "ALL_GRAINS": df.copy(),
    "GEOMETRY_SCREEN_PASS": df.loc[
        df["geometry_screen_pass"].astype(bool)
    ].copy()
}
for subset_name, frame in subsets.items():
    frame = frame.reset_index(drop=True)
    if len(frame) <= K:
        continue
    weights = build_spatial_weights(frame, K)
    for feature in features:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce"
        )
        valid = values.notna()
        # Rebuild graph if this feature contains missing values.
        if not valid.all():
            feature_frame = frame.loc[valid].reset_index(drop=True)
            feature_weights = build_spatial_weights(feature_frame, K)
            test_values = values.loc[valid].to_numpy(float)
        else:
            feature_weights = weights
            test_values = values.to_numpy(float)
        result = moran_test(test_values, feature_weights, rng)
        results.append({
            "subset": subset_name,
            "feature": feature,
            "knn_k": K,
            "permutations": PERMUTATIONS,
            **result
        })
result_df = pd.DataFrame(results)
csv_path = ROOT / "ctf_5deg_spatial_autocorrelation.csv"
json_path = ROOT / "ctf_5deg_spatial_autocorrelation_report.json"
result_df.to_csv(csv_path, index=False)
summary = {
    "dataset": "EBSD.ctf",
    "grain_count": int(len(df)),
    "geometry_screen_pass_count": int(
        df["geometry_screen_pass"].astype(bool).sum()
    ),
    "spatial_graph": {
        "method": "Symmetric union of centroid K-nearest neighbours",
        "k": K,
        "permutations": PERMUTATIONS,
        "random_seed": SEED
    },
    "results": results,
    "interpretation_guidance": [
        "Positive Moran's I indicates spatial clustering of similar feature values.",
        "Negative Moran's I indicates spatial dispersion.",
        "Permutation p-values assess spatial association under random reassignment.",
        "These tests do not establish independent experimental samples.",
        "One EBSD map cannot validate generalization to unseen specimens or sources.",
        "Multiple feature tests are exploratory; p-values are not multiplicity-adjusted."
    ],
    "ml_validation_warning": (
        "Do not use random grain-level splitting as the sole validation strategy. "
        "Future validation should group by independent map, specimen, or source."
    )
}
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("=" * 72)
print("MAT-X GRAIN SPATIAL AUTOCORRELATION AUDIT")
print("=" * 72)
print(result_df[
    [
        "subset",
        "feature",
        "n",
        "moran_i",
        "permutation_p_two_sided",
        "status"
    ]
].to_string(index=False))
print("\nSaved:", csv_path)
print("Saved:", json_path)
