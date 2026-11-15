from pathlib import Path
import json
import numpy as np
import pandas as pd
source = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Cyclic_data.dat"
)
out_dir = Path("training_results/microstructure/abaqus_deep_audit")
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(source, sep=";")
df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
df.insert(0, "original_row", np.arange(1, len(df) + 1))
numeric = ["time", "trueStress", "trueStrain", "cycl", "temp", "type"]
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("=" * 90)
print("MAT-X CYCLIC STRESS-STRAIN ANALYSIS")
print("=" * 90)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Missing values:\n", df[numeric].isna().sum().to_string())
summaries = []
for cycle, group in df.groupby("cycl", sort=True):
    group = group.sort_values("time").copy()
    t = group["time"].to_numpy()
    stress = group["trueStress"].to_numpy()
    strain = group["trueStrain"].to_numpy()
    valid = np.isfinite(t) & np.isfinite(stress) & np.isfinite(strain)
    t, stress, strain = t[valid], stress[valid], strain[valid]
    dstrain = np.diff(strain)
    reversals = int(np.sum(
        (dstrain[1:] * dstrain[:-1]) < 0
    )) if len(dstrain) > 1 else 0
    signed_area = (
        float(np.trapezoid(stress, strain))
        if len(strain) > 1 else None
    )
    summary = {
        "cycle": float(cycle),
        "measurements": int(len(group)),
        "start_time": float(t[0]) if len(t) else None,
        "end_time": float(t[-1]) if len(t) else None,
        "duration": float(t[-1] - t[0]) if len(t) > 1 else None,
        "stress_min": float(np.min(stress)) if len(stress) else None,
        "stress_max": float(np.max(stress)) if len(stress) else None,
        "strain_min": float(np.min(strain)) if len(strain) else None,
        "strain_max": float(np.max(strain)) if len(strain) else None,
        "first_stress": float(stress[0]) if len(stress) else None,
        "last_stress": float(stress[-1]) if len(stress) else None,
        "first_strain": float(strain[0]) if len(strain) else None,
        "last_strain": float(strain[-1]) if len(strain) else None,
        "stress_closure_gap": float(abs(stress[-1] - stress[0])) if len(stress) else None,
        "strain_closure_gap": float(abs(strain[-1] - strain[0])) if len(strain) else None,
        "strain_direction_reversals": reversals,
        "signed_stress_strain_integral": signed_area,
        "absolute_stress_strain_integral": abs(signed_area) if signed_area is not None else None,
    }
    summaries.append(summary)
summary_df = pd.DataFrame(summaries)
print("\nCYCLE-WISE SUMMARY")
print(summary_df.to_string(index=False))
print("\nCYCLE LABEL TRANSITIONS")
transitions = df.loc[
    df["cycl"].ne(df["cycl"].shift()),
    ["original_row", "time", "trueStress", "trueStrain", "cycl"]
]
print(transitions.to_string(index=False))
print("\nGLOBAL DATA CHECKS")
print("Cycle labels:", sorted(df["cycl"].dropna().unique().tolist()))
print("Temperature values:", sorted(df["temp"].dropna().unique().tolist()))
print("Type values:", sorted(df["type"].dropna().unique().tolist()))
print("Time strictly increasing:", bool(df["time"].is_monotonic_increasing))
print("Duplicate timestamps:", int(df["time"].duplicated().sum()))
print("Rows with missing numeric values:", int(df[numeric].isna().any(axis=1).sum()))
summary_csv = out_dir / "cyclic_cycle_summary.csv"
summary_df.to_csv(summary_csv, index=False)
transition_csv = out_dir / "cyclic_cycle_transitions.csv"
transitions.to_csv(transition_csv, index=False)
report = {
    "source_file": str(source),
    "row_count": int(len(df)),
    "column_names": list(df.columns),
    "cycle_summary": summaries,
    "cycle_transitions": transitions.to_dict(orient="records"),
    "units_verified": False,
    "cycle_completeness_verified": False,
    "original_data_modified": False,
    "interpretation": (
        "Cycle-level descriptive analysis only. "
        "Loop integrals are provisional and are not validated "
        "fatigue damage, dissipated energy, or fatigue-life measurements."
    ),
}
json_path = out_dir / "cyclic_cycle_analysis.json"
json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nREPORTS SAVED")
print(summary_csv.resolve())
print(transition_csv.resolve())
print(json_path.resolve())
print("\nCYCLIC ANALYSIS COMPLETE")
