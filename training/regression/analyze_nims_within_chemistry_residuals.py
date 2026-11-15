from pathlib import Path
import pandas as pd
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\uncertainty_calibration_v1\calibration_residual_diagnostics")
FILE = ROOT / "tensile_strength_individual_calibration_residuals.csv"
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
residuals = pd.read_csv(FILE)
dataset = pd.read_csv(DATA).reset_index(names="dataset_row_index")
nims = residuals[
    residuals["grade_id"].astype(str) == "NIMS_low_alloy_other"
].copy()
joined = nims.merge(
    dataset[["dataset_row_index", "heat_id"] + chemistry].rename(
        columns={"heat_id": "source_heat_id"}
    ),
    on="dataset_row_index",
    how="left",
    validate="one_to_one",
    indicator=True
)
print("=" * 90)
print("NIMS WITHIN-CHEMISTRY RESIDUAL ANALYSIS")
print("=" * 90)
print("Calibration records:", len(nims))
print("Matched records:", int((joined["_merge"] == "both").sum()))
print("Unmatched records:", int((joined["_merge"] != "both").sum()))
if (joined["_merge"] != "both").any():
    raise SystemExit("STOP: unmatched dataset rows.")
heat_match = (
    joined["heat_id"].astype(str) ==
    joined["source_heat_id"].astype(str)
)
print("Residual heat IDs match source heat IDs:", bool(heat_match.all()))
if not heat_match.all():
    print(joined.loc[~heat_match, ["dataset_row_index", "heat_id", "source_heat_id"]])
    raise SystemExit("STOP: heat ID mismatch.")
profiles = (
    joined.groupby(chemistry, dropna=False)
    .agg(
        heat_count=("source_heat_id", "nunique"),
        actual_min=("actual", "min"),
        actual_max=("actual", "max"),
        actual_mean=("actual", "mean"),
        actual_std=("actual", "std"),
        prediction_min=("prediction", "min"),
        prediction_max=("prediction", "max"),
        residual_min=("absolute_residual", "min"),
        residual_max=("absolute_residual", "max"),
        residual_mean=("absolute_residual", "mean"),
    )
    .reset_index()
)
profiles["actual_range"] = profiles["actual_max"] - profiles["actual_min"]
profiles["prediction_range"] = profiles["prediction_max"] - profiles["prediction_min"]
print("\nDISTINCT CHEMISTRY PROFILES:", len(profiles))
print("Profiles with exactly 3 heat IDs:", int((profiles["heat_count"] == 3).sum()))
print("\nWITHIN-PROFILE TENSILE RANGE:")
print(profiles["actual_range"].describe().to_string())
print("\nWITHIN-PROFILE PREDICTION RANGE:")
print(profiles["prediction_range"].describe().to_string())
print("\n15 PROFILES WITH LARGEST TENSILE RANGES:")
print(
    profiles.nlargest(15, "actual_range")[
        chemistry + [
            "heat_count", "actual_min", "actual_max",
            "actual_range", "prediction_range",
            "residual_min", "residual_max"
        ]
    ].to_string(index=False)
)
print("\nHEAT-LEVEL RECORDS FOR FIVE MOST VARIABLE PROFILES:")
selected = profiles.nlargest(5, "actual_range")[chemistry]
details = joined.merge(selected, on=chemistry, how="inner")
print(
    details[
        ["source_heat_id"] + chemistry +
        ["actual", "prediction", "absolute_residual"]
    ].sort_values(chemistry + ["source_heat_id"]).to_string(index=False)
)
print("\nANALYSIS COMPLETE — ORIGINAL DATA AND MODELS UNCHANGED.")
