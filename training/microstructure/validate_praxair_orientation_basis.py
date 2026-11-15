from pathlib import Path
import json
import numpy as np
import pandas as pd
base = Path("training_results/microstructure/abaqus_deep_audit")
source = base / "abaqus_praxair_material_structure_audit.json"
data = json.loads(source.read_text(encoding="utf-8"))
materials = data["materials"]
rows = []
for material in materials:
    c = np.asarray(material["constants"], dtype=float)
    # Abaqus constants are one-indexed in the report.
    # Python slices below convert them to zero-indexed positions.
    v1 = c[48:51]       # C49-C51
    v2 = c[56:59]       # C57-C59
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    dot = float(np.dot(v1, v2))
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    # Construct a right-handed orthonormal basis if v1 and v2
    # are already unit and perpendicular.
    matrix = np.column_stack((v1, v2, cross))
    determinant = float(np.linalg.det(matrix))
    rows.append({
        "material": material["name"],
        "v1_norm": n1,
        "v2_norm": n2,
        "dot_v1_v2": dot,
        "cross_norm": cross_norm,
        "basis_determinant": determinant,
        "unit_v1": abs(n1 - 1) < 1e-5,
        "unit_v2": abs(n2 - 1) < 1e-5,
        "orthogonal": abs(dot) < 1e-5,
        "valid_candidate_basis": (
            abs(n1 - 1) < 1e-5
            and abs(n2 - 1) < 1e-5
            and abs(dot) < 1e-5
        ),
        "C52_C56": ",".join(f"{x:.8g}" for x in c[51:56]),
        "C60_C64": ",".join(f"{x:.8g}" for x in c[59:64]),
        "v1": ",".join(f"{x:.9f}" for x in v1),
        "v2": ",".join(f"{x:.9f}" for x in v2),
        "cross_product": ",".join(f"{x:.9f}" for x in cross)
    })
df = pd.DataFrame(rows)
print("=" * 90)
print("PRAXAIR CANDIDATE ORIENTATION BASIS VALIDATION")
print("=" * 90)
print("\nGRAINS PASSING UNIT + ORTHOGONALITY TEST:")
print(
    df.loc[df["valid_candidate_basis"],
           ["material", "v1_norm", "v2_norm",
            "dot_v1_v2", "basis_determinant"]]
    .to_string(index=False)
)
print("\nSUMMARY:")
print("Total grains:", len(df))
print("Unit first vector:", int(df["unit_v1"].sum()))
print("Unit second vector:", int(df["unit_v2"].sum()))
print("Orthogonal vector pairs:", int(df["orthogonal"].sum()))
print("Complete candidate bases:", int(df["valid_candidate_basis"].sum()))
print("\nNORM AND DOT-PRODUCT RANGES:")
for col in ["v1_norm", "v2_norm", "dot_v1_v2",
            "cross_norm", "basis_determinant"]:
    print(
        col,
        "| min =", df[col].min(),
        "| max =", df[col].max(),
        "| mean =", df[col].mean()
    )
print("\nINTERVENING CONSTANT PATTERNS:")
print("C52-C56:")
print(df["C52_C56"].value_counts().to_string())
print("\nC60-C64:")
print(df["C60_C64"].value_counts().to_string())
csv_path = base / "abaqus_praxair_candidate_orientation_basis.csv"
json_path = base / "abaqus_praxair_candidate_orientation_basis.json"
df.to_csv(csv_path, index=False)
json_path.write_text(
    df.to_json(orient="records", indent=2),
    encoding="utf-8"
)
print("\nSaved:", csv_path)
print("Saved:", json_path)
print("\nOriginal dataset unchanged.")
