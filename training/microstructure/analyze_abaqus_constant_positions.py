from pathlib import Path
import json
import math
from collections import defaultdict
report_path = Path(
    r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure"
    r"\abaqus_deep_audit\abaqus_depvar_and_grain_material_audit.json"
)
data = json.loads(report_path.read_text(encoding="utf-8"))
print("=" * 100)
print("ABAQUS USER-MATERIAL CONSTANT POSITION ANALYSIS")
print("=" * 100)
model_vectors = {}
for model in data:
    name = model["model"]
    materials = model["materials"]
    print(f"\nMODEL: {name}")
    print("Material count:", len(materials))
    valid = [
        m for m in materials
        if len(m.get("constants", [])) == 72
    ]
    if not valid:
        print("No complete 72-constant material vectors found.")
        continue
    model_vectors[name] = {
        m["name"]: m["constants"] for m in valid
    }
    print("Complete vectors:", len(valid))
    print("DEPVAR values:", sorted({
        m.get("depvar") for m in valid
    }))
    varying = []
    constant_positions = []
    for index in range(72):
        values = [m["constants"][index] for m in valid]
        unique = sorted(set(values))
        if len(unique) > 1:
            varying.append((index + 1, len(unique), min(values), max(values)))
        else:
            constant_positions.append(index + 1)
    print("Constant positions across materials:", constant_positions)
    print("Varying positions (1-based index, unique count, min, max):")
    for item in varying:
        print(" ", item)
    if len(valid) <= 12:
        print("\nFull material vectors:")
        for material in valid:
            print(material["name"], material["constants"])
print("\n" + "=" * 100)
print("CROSS-MODEL MATERIAL VECTOR COMPARISON")
print("=" * 100)
names = list(model_vectors)
for i, name_a in enumerate(names):
    for name_b in names[i + 1:]:
        a = model_vectors[name_a]
        b = model_vectors[name_b]
        shared = set(a) & set(b)
        exact = sum(a[k] == b[k] for k in shared)
        print(
            f"{name_a} vs {name_b}: "
            f"shared material names={len(shared)}, "
            f"identical vectors={exact}/{len(shared)}"
        )
print("\n" + "=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
