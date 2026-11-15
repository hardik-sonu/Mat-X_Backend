from pathlib import Path
import re
import numpy as np
import pandas as pd
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset"
)
models = root / "Models"
cyclic_file = root / "Cyclic_data.dat"
# ---------------------------------------------------------
# 1. Robust cyclic data parsing
# ---------------------------------------------------------
print("=" * 85)
print("1. CYCLIC DATA HEADER INSPECTION")
print("=" * 85)
with cyclic_file.open("r", encoding="utf-8-sig", errors="replace") as f:
    for i in range(3):
        line = f.readline()
        print(f"RAW LINE {i+1}: {line.rstrip()}")
cyclic = pd.read_csv(
    cyclic_file,
    sep=";",
    skipinitialspace=True,
    encoding="utf-8-sig"
)
cyclic.columns = [
    str(c).strip().strip('"').strip("'").replace("\ufeff", "").lower()
    for c in cyclic.columns
]
print("\nParsed columns:", cyclic.columns.tolist())
def find_column(columns, candidates):
    for candidate in candidates:
        for col in columns:
            if col == candidate or candidate in col:
                return col
    raise KeyError(
        f"Could not find any of {candidates}. Available: {list(columns)}"
    )
time_col = find_column(cyclic.columns, ["time"])
strain_col = find_column(cyclic.columns, ["truestrain", "strain"])
print("Selected time column:", time_col)
print("Selected strain column:", strain_col)
print("Cyclic rows:", len(cyclic))
# ---------------------------------------------------------
# 2. Parse Abaqus amplitude
# ---------------------------------------------------------
amplitude_file = models / "AbaqusInput_AmpTmp" / "Amplitude.inp"
amp_lines = amplitude_file.read_text(
    encoding="utf-8-sig", errors="replace"
).splitlines()[1:]
amp_values = []
for line in amp_lines:
    if not line.strip() or line.lstrip().startswith("**"):
        continue
    values = re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",
        line
    )
    amp_values.extend(float(v) for v in values)
if len(amp_values) % 2:
    raise ValueError("Amplitude contains an odd number of numeric values.")
amp_values = np.asarray(amp_values, dtype=float).reshape(-1, 2)
print("\n" + "=" * 85)
print("2. CYCLIC DATA / AMPLITUDE COMPARISON")
print("=" * 85)
print("Amplitude pairs:", len(amp_values))
if len(cyclic) == len(amp_values):
    cyclic_time = pd.to_numeric(cyclic[time_col], errors="coerce").to_numpy()
    cyclic_strain = pd.to_numeric(cyclic[strain_col], errors="coerce").to_numpy()
    time_error = np.abs(cyclic_time - amp_values[:, 0])
    strain_error = np.abs(cyclic_strain - amp_values[:, 1])
    print("Maximum time difference:", float(np.nanmax(time_error)))
    print("Mean time difference:", float(np.nanmean(time_error)))
    print("Maximum strain difference:", float(np.nanmax(strain_error)))
    print("Mean strain difference:", float(np.nanmean(strain_error)))
    print("Time arrays identical within 1e-7:",
          bool(np.all(time_error < 1e-7)))
    print("Strain arrays identical within 1e-7:",
          bool(np.all(strain_error < 1e-7)))
else:
    print("WARNING: Different row counts; direct comparison skipped.")
# ---------------------------------------------------------
# 3. Inspect reference points and boundary conditions
# ---------------------------------------------------------
print("\n" + "=" * 85)
print("3. REFERENCE POINTS AND BOUNDARY CONDITIONS")
print("=" * 85)
for job in sorted(models.glob("Mesh*/Job_*.inp")):
    lines = job.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    print("\nMODEL:", job.parent.name)
    in_node_section = False
    reference_nodes = {}
    in_boundary = False
    boundary_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*"):
            keyword = stripped.upper()
            in_node_section = keyword.startswith("*NODE")
            in_boundary = keyword.startswith("*BOUNDARY")
            if in_boundary:
                boundary_lines.append(stripped)
            continue
        if not stripped or stripped.startswith("**"):
            continue
        if in_node_section:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    coords = [float(x) for x in parts[1:4]]
                    reference_nodes[node_id] = coords
                except ValueError:
                    pass
        elif in_boundary:
            boundary_lines.append(stripped)
    print("Job-level nodes:", reference_nodes)
    print("Boundary definitions:")
    for item in boundary_lines:
        print(" ", item)
print("\nAUDIT COMPLETE. ORIGINAL DATA AND INPUT FILES NOT MODIFIED.")
