from pathlib import Path
import re
import json
from collections import Counter
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
folder = root / "Mesh0_Main_64k"
output = (
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
    / "training_results" / "microstructure"
    / "abaqus_deep_audit"
)
output.mkdir(parents=True, exist_ok=True)
def read_materials(path):
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    materials = []
    current = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("*MATERIAL"):
            if current:
                materials.append(current)
            match = re.search(r"NAME\s*=\s*([^,]+)", line, re.I)
            current = {
                "name": match.group(1).strip() if match else "UNKNOWN",
                "depvar": None,
                "declared_constants": None,
                "constants": []
            }
        elif current and line.upper().startswith("*DEPVAR"):
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().startswith("**")
            ):
                j += 1
            if j < len(lines):
                try:
                    current["depvar"] = int(float(lines[j].strip().split(",")[0]))
                except ValueError:
                    pass
        elif current and line.upper().startswith("*USER MATERIAL"):
            match = re.search(r"CONSTANTS\s*=\s*(\d+)", line, re.I)
            current["declared_constants"] = (
                int(match.group(1)) if match else None
            )
            j = i + 1
            values = []
            while j < len(lines) and not lines[j].strip().startswith("*"):
                for token in lines[j].split(","):
                    try:
                        values.append(float(token.strip().replace("D", "E")))
                    except ValueError:
                        pass
                j += 1
            current["constants"] = values
        i += 1
    if current:
        materials.append(current)
    return materials
def read_sections(path):
    sections = []
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    for line in lines:
        if line.strip().upper().startswith("*SOLID SECTION"):
            e = re.search(r"ELSET\s*=\s*([^,]+)", line, re.I)
            m = re.search(r"MATERIAL\s*=\s*([^,]+)", line, re.I)
            sections.append({
                "elset": e.group(1).strip() if e else None,
                "material": m.group(1).strip() if m else None
            })
    return sections
material_files = list((folder / "AbaqusInput").glob("*mat*.inp"))
grain_files = list((folder / "AbaqusInput").glob("*grain*.inp"))
report = {
    "material_files": [],
    "grain_files": [],
    "materials": [],
    "sections": []
}
for path in material_files:
    materials = read_materials(path)
    report["materials"].extend(materials)
    print("\nMATERIAL FILE:", path.name)
    print("Material count:", len(materials))
    for m in materials:
        print(
            m["name"],
            "| DEPVAR:", m["depvar"],
            "| Constants:", len(m["constants"]),
            "/", m["declared_constants"]
        )
    report["material_files"].append(str(path))
for path in grain_files:
    sections = read_sections(path)
    report["sections"].extend(sections)
    print("\nSECTION FILE:", path.name)
    print("Solid sections:", len(sections))
    for s in sections[:100]:
        print(s)
    report["grain_files"].append(str(path))
counts = Counter(
    (len(m["constants"]), m["declared_constants"], m["depvar"])
    for m in report["materials"]
)
print("\nMATERIAL SIGNATURE COUNTS:")
for signature, count in counts.items():
    print(signature, ":", count)
print("\nTOTAL MATERIALS:", len(report["materials"]))
print("TOTAL SOLID SECTIONS:", len(report["sections"]))
report_path = output / "abaqus_praxair_material_structure_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nREPORT:", report_path)
print("Original dataset files were not modified.")
