from pathlib import Path
import json
import numpy as np
import pandas as pd
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
files = list(root.rglob("*.ctf"))
if len(files) != 1:
    raise RuntimeError(f"Expected one CTF file, found {len(files)}")
source = files[0]
lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
header_index = next(
    (i for i, line in enumerate(lines)
     if line.strip().startswith("Phase") and "Euler1" in line),
    None
)
if header_index is None:
    raise RuntimeError("CTF measurement header not found")
columns = lines[header_index].split()
records = []
for line_number, line in enumerate(lines[header_index + 1:], start=header_index + 2):
    parts = line.split()
    if not parts:
        continue
    if len(parts) != len(columns):
        continue
    try:
        records.append([float(x) for x in parts])
    except ValueError:
        continue
df = pd.DataFrame(records, columns=columns)
required = ["Phase", "X", "Y", "Bands", "Error", "Euler1", "Euler2", "Euler3"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[required].isna().any().any():
    raise RuntimeError("Missing or invalid values in required measurement columns")
indexed = df["Error"] == 0
euler = df.loc[indexed, ["Euler1", "Euler2", "Euler3"]]
summary = {
    "source_file": str(source),
    "file_size_bytes": source.stat().st_size,
    "total_measurement_rows": int(len(df)),
    "expected_grid_points": 272 * 272,
    "grid_count_matches": bool(len(df) == 272 * 272),
    "indexed_points_error_zero": int(indexed.sum()),
    "unindexed_points_error_nonzero": int((~indexed).sum()),
    "indexed_fraction": float(indexed.mean()),
    "phase_counts_all_points": {
        str(int(k)): int(v)
        for k, v in df["Phase"].value_counts().items()
    },
    "error_code_counts": {
        str(int(k)): int(v)
        for k, v in df["Error"].value_counts().items()
    },
    "bands_counts": {
        str(int(k)): int(v)
        for k, v in df["Bands"].value_counts().items()
    },
    "euler_angle_statistics_indexed_only": {
        col: {
            "min": float(euler[col].min()),
            "max": float(euler[col].max()),
            "mean": float(euler[col].mean()),
            "std": float(euler[col].std())
        }
        for col in euler.columns
    },
    "coordinate_grid": {
        "unique_x": int(df["X"].nunique()),
        "unique_y": int(df["Y"].nunique()),
        "x_min": float(df["X"].min()),
        "x_max": float(df["X"].max()),
        "y_min": float(df["Y"].min()),
        "y_max": float(df["Y"].max()),
        "duplicate_xy_points": int(df.duplicated(["X", "Y"]).sum())
    },
    "ctf_reference_coordinate_system": "CS0_SAMPLE_COORDINATE_SYSTEM",
    "euler_rotation_convention": "NOT_ESTABLISHED",
    "relationship_to_abaqus_coordinates": "NOT_ESTABLISHED"
}
out = Path("training_results/microstructure/abaqus_deep_audit")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "ctf_measurements_audited.csv", index=False)
(out / "ctf_measurement_audit.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 75)
print("EBSD CTF FULL MEASUREMENT AUDIT")
print("=" * 75)
print("File:", source.name)
print("Measurement rows:", len(df))
print("Expected grid points:", 272 * 272)
print("Grid count matches:", summary["grid_count_matches"])
print("Indexed points:", int(indexed.sum()))
print("Unindexed points:", int((~indexed).sum()))
print("Indexed fraction:", round(float(indexed.mean()), 6))
print("\nPHASE COUNTS:")
print(df["Phase"].value_counts().sort_index().to_string())
print("\nERROR CODE COUNTS:")
print(df["Error"].value_counts().sort_index().to_string())
print("\nBANDS COUNTS:")
print(df["Bands"].value_counts().sort_index().to_string())
print("\nEULER STATISTICS — INDEXED POINTS ONLY:")
print(euler.describe().to_string())
print("\nCOORDINATE GRID:")
print(json.dumps(summary["coordinate_grid"], indent=2))
print("\nSAVED:")
print(out / "ctf_measurements_audited.csv")
print(out / "ctf_measurement_audit.json")
print("\nOriginal CTF file unchanged.")
