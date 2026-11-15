from pathlib import Path
import pandas as pd
ROOT = Path(
    r"training_results\uncertainty_calibration_v1"
    r"\calibration_residual_diagnostics"
)
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
residuals = pd.read_csv(
    ROOT / "tensile_strength_individual_calibration_residuals.csv"
)
dataset = pd.read_csv(DATA).reset_index(names="dataset_row_index")
joined = residuals.merge(
    dataset[["dataset_row_index", "heat_id"] + chemistry],
    on="dataset_row_index",
    how="left",
    validate="one_to_one",
    suffixes=("_residual", "_source")
)
nims = joined[
    joined["grade_id"].astype(str) == "NIMS_low_alloy_other"
].copy()
if len(nims) != 81:
    raise SystemExit(f"Expected 81 records; found {len(nims)}")
if not (nims["heat_id_residual"] == nims["heat_id_source"]).all():
    raise SystemExit("STOP: heat ID mapping mismatch.")
groups = []
for profile, group in nims.groupby(chemistry, dropna=False):
    groups.append({
        **dict(zip(chemistry, profile)),
        "records": len(group),
        "heat_ids": ", ".join(group["heat_id_source"].astype(str)),
        "tensile_values": ", ".join(
            group["actual"].astype(str)
        ),
        "residuals": ", ".join(
            group["absolute_residual"].round(3).astype(str)
        ),
        "tensile_range": (
            group["actual"].max() - group["actual"].min()
        ),
        "residual_range": (
            group["absolute_residual"].max()
            - group["absolute_residual"].min()
        ),
        "mean_absolute_residual": group["absolute_residual"].mean(),
    })
profiles = pd.DataFrame(groups)
print("=" * 100)
print("NIMS REPEATED-CHEMISTRY PROFILE INVESTIGATION")
print("=" * 100)
print("Chemistry profiles:", len(profiles))
print("Records:", len(nims))
print("Records per profile:")
print(profiles["records"].value_counts().sort_index().to_string())
print("\nPROFILES WITH LARGEST TENSILE-STRENGTH RANGES:")
print(
    profiles.nlargest(10, "tensile_range")[
        ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
         "heat_ids", "tensile_values", "tensile_range",
         "residual_range", "mean_absolute_residual"]
    ].to_string(index=False)
)
print("\nPROFILES WITH LARGEST RESIDUAL RANGES:")
print(
    profiles.nlargest(10, "residual_range")[
        ["heat_ids", "tensile_values", "residuals",
         "tensile_range", "residual_range",
         "mean_absolute_residual"]
    ].to_string(index=False)
)
print("\nALL 27 PROFILE SUMMARIES:")
print(
    profiles[
        ["heat_ids", "tensile_values", "residuals",
         "tensile_range", "residual_range"]
    ].to_string(index=False)
)
print("\nAUDIT COMPLETE — NO SOURCE DATA OR MODELS MODIFIED.")
