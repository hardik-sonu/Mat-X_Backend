from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset"
)
models = root / "Models"
amplitude_path = models / "AbaqusInput_AmpTmp" / "Amplitude.inp"
data_path = root / "Cyclic_data.dat"
out = Path("training_results/microstructure/abaqus_deep_audit")
out.mkdir(parents=True, exist_ok=True)
# Read amplitude pairs
text = amplitude_path.read_text(encoding="utf-8", errors="replace")
body = "\n".join(
    line for line in text.splitlines()
    if not line.strip().startswith("*")
)
numbers = [
    float(x)
    for x in re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",
        body
    )
]
if len(numbers) % 2:
    raise ValueError(
        f"Amplitude has an odd number of numeric values: {len(numbers)}"
    )
amplitude = pd.DataFrame(
    np.array(numbers).reshape(-1, 2),
    columns=["time", "amplitude_value"]
)
data = pd.read_csv(data_path, sep=";")
data.columns = [c.strip().lstrip("#").strip() for c in data.columns]
required = {"time", "trueStress", "trueStrain", "cycl", "temp", "type"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing Cyclic_data.dat columns: {sorted(missing)}")
for col in ["time", "trueStress", "trueStrain", "cycl", "temp", "type"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=["time", "trueStress", "trueStrain", "cycl"])
# Compare amplitude signal with recorded strain at matching times
comparison = pd.merge_asof(
    data.sort_values("time"),
    amplitude.sort_values("time"),
    on="time",
    direction="nearest",
    tolerance=1e-5
)
matched = comparison.dropna(subset=["amplitude_value"]).copy()
if len(matched) >= 2:
    correlation = float(
        matched["trueStrain"].corr(matched["amplitude_value"])
    )
    strain_amplitude_mae = float(
        np.mean(
            np.abs(
                matched["trueStrain"] - matched["amplitude_value"]
            )
        )
    )
else:
    correlation = None
    strain_amplitude_mae = None
# Per-cycle descriptive statistics
cycle_rows = []
for cycle, group in data.groupby("cycl", sort=True):
    group = group.sort_values("time")
    stress = group["trueStress"].to_numpy()
    strain = group["trueStrain"].to_numpy()
    cycle_rows.append({
        "cycle": int(cycle),
        "records": int(len(group)),
        "time_start": float(group["time"].min()),
        "time_end": float(group["time"].max()),
        "duration": float(group["time"].max() - group["time"].min()),
        "stress_min": float(np.min(stress)),
        "stress_max": float(np.max(stress)),
        "stress_range": float(np.ptp(stress)),
        "strain_min": float(np.min(strain)),
        "strain_max": float(np.max(strain)),
        "strain_range": float(np.ptp(strain)),
        "temperature_min": float(group["temp"].min()),
        "temperature_max": float(group["temp"].max()),
        "first_stress": float(stress[0]),
        "last_stress": float(stress[-1]),
        "first_strain": float(strain[0]),
        "last_strain": float(strain[-1]),
        "signed_loop_integral": float(np.trapezoid(stress, strain))
            if len(group) >= 2 else None,
        "absolute_loop_integral": float(
            abs(np.trapezoid(stress, strain))
        ) if len(group) >= 2 else None,
    })
cycle_df = pd.DataFrame(cycle_rows)
# Global diagnostics
time_differences = data["time"].diff().dropna()
amplitude_time_differences = amplitude["time"].diff().dropna()
report = {
    "audit_version": "1.0.0",
    "source_files": {
        "amplitude": str(amplitude_path),
        "cyclic_data": str(data_path),
    },
    "amplitude": {
        "pair_count": int(len(amplitude)),
        "time_min": float(amplitude["time"].min()),
        "time_max": float(amplitude["time"].max()),
        "value_min": float(amplitude["amplitude_value"].min()),
        "value_max": float(amplitude["amplitude_value"].max()),
        "time_monotonic_increasing": bool(
            amplitude["time"].is_monotonic_increasing
        ),
    },
    "cyclic_data": {
        "record_count": int(len(data)),
        "time_min": float(data["time"].min()),
        "time_max": float(data["time"].max()),
        "stress_min": float(data["trueStress"].min()),
        "stress_max": float(data["trueStress"].max()),
        "strain_min": float(data["trueStrain"].min()),
        "strain_max": float(data["trueStrain"].max()),
        "temperature_values": sorted(
            data["temp"].dropna().unique().tolist()
        ),
        "cycle_counts": {
            str(int(k)): int(v)
            for k, v in data.groupby("cycl").size().items()
        },
        "time_monotonic_increasing": bool(
            data["time"].is_monotonic_increasing
        ),
        "duplicate_time_records": int(data["time"].duplicated().sum()),
        "minimum_time_increment": float(time_differences.min())
            if len(time_differences) else None,
        "maximum_time_increment": float(time_differences.max())
            if len(time_differences) else None,
    },
    "amplitude_strain_comparison": {
        "matched_records": int(len(matched)),
        "correlation": correlation,
        "mean_absolute_difference": strain_amplitude_mae,
        "interpretation": (
            "Numerical comparison only; physical equivalence is not established."
        ),
    },
    "cycle_analysis": cycle_rows,
    "limitations": [
        "No units inferred for amplitude, stress, strain, or time.",
        "No experimental or simulation provenance inferred for Cyclic_data.dat.",
        "Cycle labels are retained exactly as supplied.",
        "A partial final cycle is not treated as a complete fatigue cycle.",
        "Loop integrals are descriptive numerical integrals, not validated fatigue damage."
    ],
}
cycle_path = out / "abaqus_cyclic_cycle_analysis.csv"
comparison_path = out / "abaqus_amplitude_strain_comparison.csv"
report_path = out / "abaqus_cyclic_loading_audit.json"
cycle_df.to_csv(cycle_path, index=False)
comparison.to_csv(comparison_path, index=False)
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("=" * 90)
print("ABAQUS CYCLIC LOADING AUDIT")
print("=" * 90)
print("\nAMPLITUDE")
print(json.dumps(report["amplitude"], indent=2))
print("\nCYCLIC DATA")
print(json.dumps(report["cyclic_data"], indent=2))
print("\nAMPLITUDE VS TRUE STRAIN")
print(json.dumps(report["amplitude_strain_comparison"], indent=2))
print("\nCYCLE ANALYSIS")
print(cycle_df.to_string(index=False))
print("\nOUTPUT FILES")
print(cycle_path.resolve())
print(comparison_path.resolve())
print(report_path.resolve())
print("\nAUDIT COMPLETE. ORIGINAL DATA WAS NOT MODIFIED.")
