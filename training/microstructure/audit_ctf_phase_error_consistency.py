from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.ndimage import label
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
source = next(root.rglob("*.ctf"))
lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
header = next(i for i, line in enumerate(lines)
              if line.strip().startswith("Phase") and "Euler1" in line)
columns = lines[header].split()
rows = []
for line in lines[header + 1:]:
    parts = line.split()
    if len(parts) != len(columns):
        continue
    try:
        rows.append([float(x) for x in parts])
    except ValueError:
        continue
df = pd.DataFrame(rows, columns=columns)
print("=" * 75)
print("CTF PHASE / ERROR CONSISTENCY AUDIT")
print("=" * 75)
cross = pd.crosstab(df["Phase"], df["Error"])
print("\nPHASE × ERROR COUNTS:")
print(cross.to_string())
suspect = df[(df["Phase"] == 0) & (df["Error"] == 0)].copy()
print("\nPHASE 0 / ERROR 0 RECORDS:", len(suspect))
if len(suspect):
    selected = ["X", "Y", "Bands", "Error", "Euler1", "Euler2",
                "Euler3", "MAD", "BC", "BS"]
    print(suspect[selected].to_string(index=False))
    print("\nPHASE 0 / ERROR 0 SUMMARY:")
    print(
        suspect[["Bands", "Euler1", "Euler2", "Euler3", "MAD", "BC"]]
        .describe()
        .to_string()
    )
nx = int(df["X"].nunique())
ny = int(df["Y"].nunique())
phase0_good = (
    ((df["Phase"] == 0) & (df["Error"] == 0))
    .to_numpy()
    .reshape(ny, nx)
)
structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
labels, count = label(phase0_good, structure=structure)
sizes = np.bincount(labels.ravel())[1:]
print("\nPHASE 0 / ERROR 0 CONNECTED COMPONENTS:", int(count))
print("COMPONENT SIZES:", sizes.tolist())
print("\nBOUNDARY DISTRIBUTION:")
print("Top row:", int(phase0_good[0, :].sum()))
print("Bottom row:", int(phase0_good[-1, :].sum()))
print("Left column:", int(phase0_good[:, 0].sum()))
print("Right column:", int(phase0_good[:, -1].sum()))
report = {
    "total_rows": int(len(df)),
    "phase_error_cross_tab": {
        str(p): {str(e): int(v) for e, v in row.items()}
        for p, row in cross.iterrows()
    },
    "phase0_error0_count": int(len(suspect)),
    "phase0_error0_component_count": int(count),
    "phase0_error0_component_sizes": sizes.tolist(),
    "phase0_error0_records": suspect.to_dict(orient="records")
}
out = Path("training_results/microstructure/abaqus_deep_audit")
out.mkdir(parents=True, exist_ok=True)
(out / "ctf_phase_error_consistency_audit.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("\nSaved:", out / "ctf_phase_error_consistency_audit.json")
print("Original CTF unchanged.")
