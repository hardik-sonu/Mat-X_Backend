from pathlib import Path
import json
import numpy as np
path = Path(
    r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure"
    r"\abaqus_deep_audit\abaqus_depvar_and_grain_material_audit.json"
)
data = json.loads(path.read_text(encoding="utf-8"))
print("=" * 100)
print("GRAIN ORIENTATION MATHEMATICAL AUDIT")
print("=" * 100)
for model in data:
    if model["model"].startswith("Mesh0"):
        continue
    print("\nMODEL:", model["model"])
    for material in model["materials"]:
        c = np.asarray(material["constants"], dtype=float)
        if len(c) != 72:
            continue
        # Candidate triplets, using 1-based positions.
        a = c[48:51]
        b = c[57:60]
        d = c[64:67]
        vectors = [a, b, d]
        norms = [np.linalg.norm(v) for v in vectors]
        normalized = [
            v / n if n > 1e-12 else v
            for v, n in zip(vectors, norms)
        ]
        dots = [
            float(np.dot(normalized[0], normalized[1])),
            float(np.dot(normalized[0], normalized[2])),
            float(np.dot(normalized[1], normalized[2]))
        ]
        matrix = np.column_stack(normalized)
        determinant = float(np.linalg.det(matrix))
        print(f"\n{material['name']}")
        print("Positions 49-51:", a.tolist())
        print("Positions 58-60:", b.tolist())
        print("Positions 65-67:", d.tolist())
        print("Vector norms:", [round(x, 8) for x in norms])
        print("Normalized pairwise dot products:",
              [round(x, 8) for x in dots])
        print("Candidate matrix determinant:", round(determinant, 8))
        print("Orthogonal within 1e-5:",
              all(abs(x) < 1e-5 for x in dots))
        print("Unit vectors within 1e-5:",
              all(abs(x - 1) < 1e-5 for x in norms))
print("\nAUDIT COMPLETE. NO ORIGINAL FILES MODIFIED.")
