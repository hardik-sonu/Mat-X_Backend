from pathlib import Path
import re
import json
import hashlib
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
output = (
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
    / "training_results" / "microstructure" / "abaqus_deep_audit"
)
output.mkdir(parents=True, exist_ok=True)
def numeric_values(lines):
    values = []
    for line in lines:
        if line.strip().startswith("**"):
            continue
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                pass
    return values
def parse_material_file(path):
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    materials = []
    current = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()
        if upper.startswith("*MATERIAL"):
            if current:
                materials.append(current)
            match = re.search(r"NAME\s*=\s*([^,]+)", line, re.I)
            current = {
                "name": match.group(1).strip() if match else "UNKNOWN",
                "depvar": None,
                "user_material_declared": None,
                "constants": []
            }
        elif current and upper.startswith("*DEPVAR"):
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].strip().startswith("**")
            ):
                j += 1
            if j < len(lines):
                vals = numeric_values([lines[j]])
                current["depvar"] = vals[0] if vals else None
        elif current and upper.startswith("*USER MATERIAL"):
            match = re.search(r"CONSTANTS\s*=\s*(\d+)", line, re.I)
            current["user_material_declared"] = (
                int(match.group(1)) if match else None
            )
            j = i + 1
            data_lines = []
            while j < len(lines) and not lines[j].strip().startswith("*"):
                data_lines.append(lines[j])
                j += 1
            current["constants"] = numeric_values(data_lines)
        i += 1
    if current:
        materials.append(current)
    return materials
def parse_sections(path):
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    sections = []
    for line in lines:
        if line.strip().upper().startswith("*SOLID SECTION"):
            elset = re.search(r"ELSET\s*=\s*([^,]+)", line, re.I)
            material = re.search(r"MATERIAL\s*=\s*([^,]+)", line, re.I)
            sections.append({
                "elset": elset.group(1).strip() if elset else None,
                "material": material.group(1).strip() if material else None
            })
    return sections
report = []
for folder in sorted(root.glob("Mesh*")):
    mat_file = folder / "AbaqusInput" / "Hom_316L_mat.inp"
    grain_file = folder / "AbaqusInput" / "Hom_316L_grains.inp"
    if not mat_file.exists():
        continue
    materials = parse_material_file(mat_file)
    sections = parse_sections(grain_file) if grain_file.exists() else []
    model = {
        "model": folder.name,
        "materials": materials,
        "solid_sections": sections
    }
    report.append(model)
    print("\n" + "=" * 90)
    print("MODEL:", folder.name)
    print("=" * 90)
    print("Material definitions:", len(materials))
    print("Solid sections:", len(sections))
    for material in materials:
        constants = material["constants"]
        digest = hashlib.sha256(
            ",".join(f"{x:.12g}" for x in constants).encode()
        ).hexdigest()[:16]
        print(
            f'{material["name"]:12} '
            f'DEPVAR={material["depvar"]} '
            f'CONSTANTS={len(constants)}/'
            f'{material["user_material_declared"]} '
            f'HASH={digest}'
        )
        print("  First 12 constants:", constants[:12])
    print("Section assignments:")
    for section in sections:
        print(" ", section)
report_path = output / "abaqus_depvar_and_grain_material_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nREPORT:", report_path)
print("AUDIT COMPLETE. ORIGINAL FILES NOT MODIFIED.")
