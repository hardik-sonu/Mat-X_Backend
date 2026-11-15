from pathlib import Path
import re, json, csv, hashlib
from collections import Counter
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure\abaqus_deep_audit")
OUT.mkdir(parents=True, exist_ok=True)
def read_text(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return path.read_text(errors="replace")
def parse_numbers(line):
    vals = []
    for x in line.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return vals
def keyword(line):
    s = line.strip()
    return s.startswith("*") and not s.startswith("**")
def parse_deck(path):
    lines = read_text(path).splitlines()
    sections = []
    current = None
    for i, line in enumerate(lines, 1):
        if keyword(line):
            current = {
                "keyword": line.strip(),
                "line": i,
                "data": []
            }
            sections.append(current)
        elif current and line.strip() and not line.lstrip().startswith("**"):
            current["data"].append(line.strip())
    counts = Counter()
    for sec in sections:
        name = sec["keyword"].split(",")[0].strip("*").upper()
        counts[name] += 1
    nodes = 0
    elements = 0
    element_types = Counter()
    materials = []
    includes = []
    steps = []
    boundary_data = []
    amplitudes = []
    user_materials = []
    orientations = []
    for sec in sections:
        kw = sec["keyword"]
        upper = kw.upper()
        if upper.startswith("*NODE"):
            nodes += sum(
                1 for row in sec["data"]
                if len(parse_numbers(row)) >= 3
            )
        if upper.startswith("*ELEMENT"):
            match = re.search(r"TYPE\s*=\s*([^,\s]+)", kw, re.I)
            etype = match.group(1) if match else "UNKNOWN"
            element_types[etype] += sum(
                1 for row in sec["data"]
                if len(parse_numbers(row)) >= 3
            )
            elements += sum(
                1 for row in sec["data"]
                if len(parse_numbers(row)) >= 3
            )
        if upper.startswith("*MATERIAL"):
            match = re.search(r"NAME\s*=\s*([^,]+)", kw, re.I)
            materials.append({
                "name": match.group(1).strip() if match else "UNKNOWN",
                "line": sec["line"]
            })
        if upper.startswith("*INCLUDE"):
            match = re.search(r"INPUT\s*=\s*([^,\s]+)", kw, re.I)
            if match:
                target = (path.parent / match.group(1).replace("\\", "/")).resolve()
                includes.append({
                    "reference": match.group(1),
                    "resolved_path": str(target),
                    "exists": target.exists()
                })
        if upper.startswith("*STEP"):
            steps.append({
                "line": sec["line"],
                "definition": kw
            })
        if upper.startswith("*BOUNDARY"):
            boundary_data.append({
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
        if upper.startswith("*AMPLITUDE"):
            amplitudes.append({
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
        if upper.startswith("*USER MATERIAL"):
            user_materials.append({
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"],
                "numeric_values": [
                    n for row in sec["data"] for n in parse_numbers(row)
                ]
            })
        if upper.startswith("*ORIENTATION"):
            orientations.append({
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
    material_blocks = []
    for idx, mat in enumerate(materials):
        start = mat["line"] - 1
        end = materials[idx + 1]["line"] - 1 if idx + 1 < len(materials) else len(lines)
        block = lines[start:end]
        material_blocks.append({
            "name": mat["name"],
            "line": mat["line"],
            "keywords": [
                x.strip() for x in block if keyword(x)
            ],
            "density_values": [
                x.strip() for x in block
                if re.match(r"\s*\d", x)
            ][:4]
        })
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "line_count": len(lines),
        "sha256": digest,
        "keyword_counts": dict(counts),
        "node_rows": nodes,
        "element_rows": elements,
        "element_types": dict(element_types),
        "materials": material_blocks,
        "material_count": len(materials),
        "include_references": includes,
        "missing_includes": [x for x in includes if not x["exists"]],
        "simulation_steps": steps,
        "boundary_conditions": boundary_data,
        "amplitudes": amplitudes,
        "user_materials": user_materials,
        "orientations": orientations
    }
files = sorted(ROOT.rglob("*.inp"))
results = []
for path in files:
    try:
        results.append(parse_deck(path))
    except Exception as exc:
        results.append({
            "file": str(path),
            "error": repr(exc)
        })
# Focus on the grain-orientation and cyclic-behavior dataset.
focused = [
    x for x in results
    if "Grain orientation and cyclic behavior" in x.get("file", "")
]
# Compare mesh configurations using main job decks.
mesh_rows = []
for item in focused:
    name = Path(item["file"]).name
    if name.lower().startswith("job_"):
        mesh_rows.append({
            "file": item["file"],
            "size_bytes": item["size_bytes"],
            "nodes_in_main_deck": item.get("node_rows", 0),
            "elements_in_main_deck": item.get("element_rows", 0),
            "element_types": item.get("element_types", {}),
            "materials": item.get("material_count", 0),
            "steps": len(item.get("simulation_steps", [])),
            "includes": len(item.get("include_references", [])),
            "missing_includes": len(item.get("missing_includes", []))
        })
# Save complete JSON.
report = {
    "dataset_root": str(ROOT),
    "total_inp_files": len(results),
    "focused_files": len(focused),
    "mesh_job_decks": mesh_rows,
    "files": results,
    "interpretation": {
        "evidence_type": "COMPUTATIONAL_FINITE_ELEMENT_INPUT",
        "experimental_measurement": False,
        "simulation_results_present": False,
        "user_material_subroutine_verified": False,
        "units_verified": False
    }
}
(OUT / "abaqus_deep_audit.json").write_text(
    json.dumps(report, indent=2, default=str),
    encoding="utf-8"
)
with (OUT / "abaqus_mesh_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    if mesh_rows:
        writer = csv.DictWriter(f, fieldnames=list(mesh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mesh_rows)
summary = [
    "MAT-X ABAQUS DEEP AUDIT",
    "=" * 80,
    f"All INP files: {len(results)}",
    f"Focused dataset files: {len(focused)}",
    f"Main job decks: {len(mesh_rows)}",
    ""
]
for item in focused:
    if "error" in item:
        summary.append(f"ERROR: {item['file']} | {item['error']}")
        continue
    summary += [
        "-" * 80,
        item["file"],
        f"Lines: {item['line_count']}",
        f"Materials: {item['material_count']}",
        f"Node rows in this file: {item['node_rows']}",
        f"Element rows in this file: {item['element_rows']}",
        f"Element types: {item['element_types']}",
        f"User material blocks: {len(item['user_materials'])}",
        f"Steps: {item['simulation_steps']}",
        f"Missing includes: {item['missing_includes']}"
    ]
    for mat in item["materials"]:
        summary.append(f"Material: {mat['name']} | line {mat['line']}")
    for um in item["user_materials"]:
        summary.append(
            f"User material line {um['line']}: "
            f"{um['keyword']} | numeric values={len(um['numeric_values'])}"
        )
    for inc in item["include_references"]:
        summary.append(
            f"INCLUDE: {inc['reference']} | exists={inc['exists']}"
        )
summary += [
    "",
    "MESH JOB COMPARISON",
    "=" * 80
]
for row in mesh_rows:
    summary.append(json.dumps(row, default=str))
summary += [
    "",
    "CAUTION:",
    "Input decks are computational evidence, not experimental measurements.",
    "No Abaqus simulations were executed by this audit.",
    "User-material constants and units remain uninterpreted pending source documentation."
]
(OUT / "abaqus_deep_audit.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)
print("\n" + "=" * 80)
print("ABAQUS DEEP AUDIT COMPLETE")
print("=" * 80)
print("INP files scanned:", len(results))
print("Focused files:", len(focused))
print("Main job decks:", len(mesh_rows))
print("\nMESH COMPARISON:")
for row in mesh_rows:
    print(row)
print("\nOUTPUT:")
print(OUT)
print(" - abaqus_deep_audit.json")
print(" - abaqus_mesh_comparison.csv")
print(" - abaqus_deep_audit.txt")
