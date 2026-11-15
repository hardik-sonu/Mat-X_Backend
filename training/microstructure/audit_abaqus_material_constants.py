from pathlib import Path
import re
import json
import hashlib
import numpy as np
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
output = (
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
    / "training_results"
    / "microstructure"
    / "abaqus_deep_audit"
)
output.mkdir(parents=True, exist_ok=True)
def parse_material_file(path):
    lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    materials = []
    current_material = None
    mode = None
    for line_number, line in enumerate(lines, start=1):
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            upper = s.upper()
            if upper.startswith("*MATERIAL"):
                match = re.search(
                    r"name\s*=\s*([^,\s]+)", s, re.I
                )
                current_material = {
                    "name": match.group(1) if match else "UNKNOWN",
                    "line": line_number,
                    "user_material_constants_declared": None,
                    "constants": [],
                    "depvar_declared": None,
                    "depvar_line": None
                }
                materials.append(current_material)
                mode = None
            elif upper.startswith("*USER MATERIAL"):
                if current_material is not None:
                    match = re.search(
                        r"constants\s*=\s*(\d+)", s, re.I
                    )
                    current_material[
                        "user_material_constants_declared"
                    ] = int(match.group(1)) if match else None
                    mode = "constants"
                else:
                    mode = None
            elif upper.startswith("*DEPVAR"):
                if current_material is not None:
                    match = re.search(r"\d+", s)
                    current_material["depvar_declared"] = (
                        int(match.group(0)) if match else None
                    )
                    current_material["depvar_line"] = line_number
                mode = None
            else:
                mode = None
            continue
        if mode == "constants" and current_material is not None:
            tokens = [x.strip() for x in s.split(",") if x.strip()]
            for token in tokens:
                try:
                    current_material["constants"].append(float(token))
                except ValueError:
                    current_material["constants"].append(
                        {"unparsed": token, "line": line_number}
                    )
    for material in materials:
        values = material["constants"]
        numeric = [
            x for x in values
            if isinstance(x, (int, float))
        ]
        material["constant_values_parsed"] = len(values)
        material["constant_numeric_count"] = len(numeric)
        material["constant_count_matches_declaration"] = (
            len(values) == material["user_material_constants_declared"]
            if material["user_material_constants_declared"] is not None
            else None
        )
        if len(numeric) == len(values) and numeric:
            raw = ",".join(f"{x:.12g}" for x in numeric)
            material["constants_sha256"] = hashlib.sha256(
                raw.encode()
            ).hexdigest()
            material["first_12_constants"] = numeric[:12]
            material["last_6_constants"] = numeric[-6:]
            material["constant_min"] = min(numeric)
            material["constant_max"] = max(numeric)
            material["constant_mean"] = float(np.mean(numeric))
        else:
            material["constants_sha256"] = None
    return materials
def scan_assignments(path):
    lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    counts = {
        "material_keywords": 0,
        "solid_sections": 0,
        "solid_section_material_names": [],
        "elsets": 0,
        "orientations": 0,
        "distributions": 0,
        "depvar_keywords": 0,
        "user_material_keywords": 0
    }
    for line in lines:
        s = line.strip()
        if not s.startswith("*"):
            continue
        u = s.upper()
        if u.startswith("*MATERIAL"):
            counts["material_keywords"] += 1
        elif u.startswith("*SOLID SECTION"):
            counts["solid_sections"] += 1
            match = re.search(r"material\s*=\s*([^,\s]+)", s, re.I)
            if match:
                counts["solid_section_material_names"].append(
                    match.group(1)
                )
        elif u.startswith("*ELSET"):
            counts["elsets"] += 1
        elif u.startswith("*ORIENTATION"):
            counts["orientations"] += 1
        elif u.startswith("*DISTRIBUTION"):
            counts["distributions"] += 1
        elif u.startswith("*DEPVAR"):
            counts["depvar_keywords"] += 1
        elif u.startswith("*USER MATERIAL"):
            counts["user_material_keywords"] += 1
    return counts
all_models = []
all_hashes = {}
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir() or not model_dir.name.startswith("Mesh"):
        continue
    material_files = list(
        (model_dir / "AbaqusInput").glob("*mat*.inp")
    )
    job_files = list(model_dir.glob("Job_*.inp"))
    model_report = {
        "model": model_dir.name,
        "material_files": [],
        "job_files": []
    }
    for path in material_files:
        materials = parse_material_file(path)
        model_report["material_files"].append({
            "file": str(path.relative_to(root)),
            "material_count": len(materials),
            "materials": materials
        })
        for material in materials:
            digest = material["constants_sha256"]
            if digest:
                all_hashes.setdefault(digest, []).append(
                    f"{model_dir.name}:{material['name']}"
                )
    for path in job_files:
        model_report["job_files"].append({
            "file": str(path.relative_to(root)),
            **scan_assignments(path)
        })
    all_models.append(model_report)
print("=" * 100)
print("ABAQUS MATERIAL-CONSTANT AND ELEMENT-ASSIGNMENT AUDIT")
print("=" * 100)
for model in all_models:
    print("\nMODEL:", model["model"])
    for file_report in model["material_files"]:
        print(
            "MATERIAL FILE:",
            file_report["file"],
            "| definitions:",
            file_report["material_count"]
        )
        for mat in file_report["materials"]:
            print(
                f'  {mat["name"]}: '
                f'declared={mat["user_material_constants_declared"]}, '
                f'parsed={mat["constant_values_parsed"]}, '
                f'count_match={mat["constant_count_matches_declaration"]}, '
                f'DEPVAR={mat["depvar_declared"]}, '
                f'min={mat.get("constant_min")}, '
                f'max={mat.get("constant_max")}, '
                f'hash={str(mat["constants_sha256"])[:16]}'
            )
            print("    First 12:", mat.get("first_12_constants"))
    for job in model["job_files"]:
        print("JOB:", job["file"])
        print(
            "  Materials:", job["material_keywords"],
            "| User Material:", job["user_material_keywords"],
            "| Solid Sections:", job["solid_sections"],
            "| ELSETs:", job["elsets"],
            "| Orientations:", job["orientations"],
            "| Distributions:", job["distributions"],
            "| DEPVAR:", job["depvar_keywords"]
        )
        print(
            "  Solid-section material assignments:",
            job["solid_section_material_names"]
        )
print("\n" + "=" * 100)
print("CONSTANT-VECTOR UNIQUENESS")
print("=" * 100)
print("Unique constant vectors:", len(all_hashes))
for digest, occurrences in sorted(
    all_hashes.items(), key=lambda x: (-len(x[1]), x[0])
):
    print(
        "Occurrences:", len(occurrences),
        "| SHA256:", digest[:20],
        "| Examples:", occurrences[:5]
    )
report = {
    "models": all_models,
    "unique_constant_vector_count": len(all_hashes),
    "constant_vector_occurrences": all_hashes,
    "interpretation_limit": (
        "Numerical structure only. Physical meaning of user material "
        "constants cannot be established without the UMAT source or "
        "authoritative constitutive documentation."
    )
}
report_path = output / "abaqus_material_constants_assignment_audit.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("\nJSON REPORT:", report_path)
print("\nAUDIT COMPLETE. ORIGINAL INPUT FILES NOT MODIFIED.")
