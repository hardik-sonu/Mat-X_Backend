from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import Counter
base = Path("training_results/microstructure/abaqus_deep_audit")
praxair_path = base / "abaqus_praxair_material_structure_audit.json"
grain_path = base / "abaqus_depvar_and_grain_material_audit.json"
praxair = json.loads(praxair_path.read_text(encoding="utf-8"))
materials = praxair["materials"]
rows = []
for material in materials:
    row = {"material": material["name"]}
    for i, value in enumerate(material["constants"], start=1):
        row[f"C{i:02d}"] = value
    rows.append(row)
df = pd.DataFrame(rows)
constant_cols = [c for c in df.columns if c.startswith("C")]
print("=" * 100)
print("PRAXAIR 75-GRAIN CONSTANT VARIATION AUDIT")
print("=" * 100)
print("\nMaterial count:", len(df))
print("Constant columns:", len(constant_cols))
variation = []
for col in constant_cols:
    values = pd.to_numeric(df[col], errors="coerce")
    unique = values.nunique(dropna=True)
    variation.append({
        "constant": col,
        "unique_values": int(unique),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=0)),
        "varies_between_grains": bool(unique > 1)
    })
variation_df = pd.DataFrame(variation)
print("\nCONSTANTS SHARED BY ALL 75 GRAINS:")
print(
    variation_df.loc[
        ~variation_df["varies_between_grains"]
    ].to_string(index=False)
)
print("\nCONSTANTS THAT VARY:")
print(
    variation_df.loc[
        variation_df["varies_between_grains"]
    ].to_string(index=False)
)
vectors = [
    tuple(float(x) for x in material["constants"])
    for material in materials
]
counter = Counter(vectors)
print("\nUNIQUE COMPLETE MATERIAL VECTORS:", len(counter))
print("DUPLICATE VECTOR GROUPS:", sum(n > 1 for n in counter.values()))
print("\nFIRST FIVE MATERIAL DEFINITIONS:")
for material in materials[:5]:
    print(
        material["name"],
        "DEPVAR =", material["depvar"],
        "CONSTANTS =", len(material["constants"])
    )
    print(material["constants"])
# Inspect the same candidate triplet positions previously examined.
triplets = {
    "C49_C51": (48, 51),
    "C58_C60": (57, 60),
    "C65_C67": (64, 67)
}
print("\nCANDIDATE TRIPLET STATISTICS:")
for name, (start, end) in triplets.items():
    values = np.array([
        material["constants"][start:end]
        for material in materials
    ], dtype=float)
    norms = np.linalg.norm(values, axis=1)
    print(
        name,
        "| norm min/max:",
        round(float(norms.min()), 8),
        round(float(norms.max()), 8),
        "| mean norm:",
        round(float(norms.mean()), 8)
    )
# Save detailed audit files.
variation_path = base / "abaqus_praxair_constant_variation.csv"
materials_path = base / "abaqus_praxair_material_constant_matrix.csv"
variation_df.to_csv(variation_path, index=False)
df.to_csv(materials_path, index=False)
print("\nSAVED:")
print(variation_path)
print(materials_path)
print("\nAUDIT COMPLETE. ORIGINAL DATASET UNCHANGED.")
