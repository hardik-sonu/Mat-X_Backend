from pathlib import Path
import math
import numpy as np
import pandas as pd
ROOT = Path(
    r"training_results\uncertainty_calibration_v1"
    r"\calibration_residual_diagnostics"
)
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
FILE = ROOT / "tensile_strength_individual_calibration_residuals.csv"
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
residuals = pd.read_csv(FILE)
dataset = pd.read_csv(DATA).reset_index(names="dataset_row_index")
required = [
    "dataset_row_index", "grade_id", "actual",
    "prediction", "absolute_residual"
]
missing = [c for c in required if c not in residuals.columns]
if missing:
    raise SystemExit(f"Residual file missing columns: {missing}")
joined = residuals.merge(
    dataset[["dataset_row_index", "heat_id"] + chemistry].rename(
        columns={"heat_id": "source_heat_id"}
    ),
    on="dataset_row_index",
    how="left",
    validate="one_to_one",
    indicator=True
)
if not (joined["_merge"] == "both").all():
    raise SystemExit("STOP: incomplete dataset-row mapping.")
nims = joined[
    joined["grade_id"].astype(str) == "NIMS_low_alloy_other"
].copy()
other = joined[
    joined["grade_id"].astype(str) != "NIMS_low_alloy_other"
].copy()
if len(nims) != 81:
    raise SystemExit(f"Expected 81 NIMS records, found {len(nims)}")
if not (nims["heat_id"].astype(str) == nims["source_heat_id"].astype(str)).all():
    raise SystemExit("STOP: residual heat IDs do not match source heat IDs.")
profile_counts = nims.groupby(chemistry, dropna=False).size()
if len(profile_counts) != 27 or not (profile_counts == 3).all():
    raise SystemExit("STOP: expected 27 chemistry profiles with 3 records each.")
if nims["source_heat_id"].nunique() != 81:
    raise SystemExit("STOP: NIMS heat IDs are not unique.")
def conformal_quantile(scores, coverage):
    values = np.sort(np.asarray(scores, dtype=float))
    n = len(values)
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return float("inf"), rank, n
    return float(values[rank - 1]), rank, n
other_scores = (
    other.groupby("grade_id")["absolute_residual"]
    .max()
    .to_numpy(dtype=float)
)
original_nims_score = float(nims["absolute_residual"].max())
profile_scores = (
    nims.groupby(chemistry, dropna=False)["absolute_residual"]
    .max()
    .to_numpy(dtype=float)
)
heat_scores = (
    nims.groupby("source_heat_id")["absolute_residual"]
    .max()
    .to_numpy(dtype=float)
)
scenarios = {
    "ORIGINAL_GRADE_GROUP": np.concatenate([
        other_scores, [original_nims_score]
    ]),
    "NIMS_CHEMISTRY_PROFILE_GROUPS": np.concatenate([
        other_scores, profile_scores
    ]),
    "NIMS_HEAT_LEVEL_GROUPS": np.concatenate([
        other_scores, heat_scores
    ]),
}
print("=" * 95)
print("NIMS GROUPING SENSITIVITY — TENSILE CALIBRATION")
print("=" * 95)
print("Non-NIMS grade groups:", len(other_scores))
print("NIMS records:", len(nims))
print("NIMS chemistry profiles:", len(profile_scores))
print("NIMS unique heat IDs:", len(heat_scores))
print("Original NIMS maximum:", round(original_nims_score, 6))
rows = []
for name, scores in scenarios.items():
    q90, rank90, n90 = conformal_quantile(scores, 0.90)
    q95, rank95, n95 = conformal_quantile(scores, 0.95)
    rows.append({
        "scenario": name,
        "number_of_groups": len(scores),
        "q90": q90,
        "q90_rank": rank90,
        "q95": q95,
        "q95_rank": rank95,
    })
print("\nQUANTILE COMPARISON:")
print(pd.DataFrame(rows).to_string(index=False))
print("\nINTERPRETATION:")
print(
    "This is a grouping sensitivity analysis, not a validated "
    "alternative conformal calibration. The alternative grouping "
    "units have not been established as exchangeable."
)
print("\nAUDIT COMPLETE — ORIGINAL DATA AND MODELS UNCHANGED.")
