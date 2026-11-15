from pathlib import Path
from collections import Counter
import re, json, csv, hashlib
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
DATASET = ROOT / "Grain orientation and cyclic behavior" / "Dataset" / "Models"
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure\abaqus_deep_audit")
OUT.mkdir(parents=True, exist_ok=True)
def read_lines(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except UnicodeDecodeError:
            pass
    return path.read_text(errors="replace").splitlines()
def is_keyword(line):
    s = line.strip()
    return s.startswith("*") and not s.startswith("**")
def sections(path):
    lines = read_lines(path)
    result = []
    current = None
    for number, line in enumerate(lines, 1):
        s = line.strip()
        if is_keyword(line):
            current = {
                "keyword": s,
                "line": number,
                "data": []
            }
            result.append(current)
        elif current and s and not s.startswith("**"):
            current["data"].append(s)
    return result
def keyword_name(text):
    return text.split(",")[0].strip("* ").upper()
def get_attribute(text, name):
    match = re.search(
        rf"(?:^|,)\s*{re.escape(name)}\s*=\s*([^,]+)",
        text,
        re.I
    )
    return match.group(1).strip().strip('"') if match else None
def numeric_tokens(line):
    values = []
    for token in line.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return values
def resolve_include(parent, reference):
    ref = reference.replace("\\", "/")
    return (parent / ref).resolve()
def audit_model(main_file):
    visited = set()
    file_records = []
    all_sections = []
    missing = []
    errors = []
    def visit(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        if not path.exists():
            missing.append(str(path))
            return
        try:
            parsed = sections(path)
            file_records.append({
                "path": str(path),
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            })
            for sec in parsed:
                sec_copy = dict(sec)
                sec_copy["source_file"] = str(path.relative_to(ROOT))
                all_sections.append(sec_copy)
                if keyword_name(sec["keyword"]) == "INCLUDE":
                    reference = get_attribute(sec["keyword"], "INPUT")
                    if reference:
                        visit(resolve_include(path.parent, reference))
                    else:
                        errors.append({
                            "file": str(path),
                            "line": sec["line"],
                            "error": "INCLUDE keyword has no INPUT attribute"
                        })
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})
    visit(main_file)
    counts = Counter(keyword_name(s["keyword"]) for s in all_sections)
    node_rows = []
    element_rows = []
    element_types = Counter()
    materials = []
    user_materials = []
    density_blocks = []
    orientations = []
    steps = []
    boundary_conditions = []
    amplitudes = []
    loads = []
    initial_conditions = []
    output_requests = []
    sections_info = []
    for sec in all_sections:
        kw = sec["keyword"]
        name = keyword_name(kw)
        source = sec["source_file"]
        if name == "NODE":
            valid = [
                row for row in sec["data"]
                if len(numeric_tokens(row)) >= 4
            ]
            node_rows.extend({
                "file": source,
                "line": sec["line"],
                "row": row
            } for row in valid)
        elif name == "ELEMENT":
            etype = get_attribute(kw, "TYPE") or "UNKNOWN"
            valid = [
                row for row in sec["data"]
                if len(numeric_tokens(row)) >= 3
            ]
            element_types[etype] += len(valid)
            element_rows.extend({
                "file": source,
                "line": sec["line"],
                "type": etype,
                "row": row
            } for row in valid)
        elif name == "MATERIAL":
            materials.append({
                "name": get_attribute(kw, "NAME") or "UNKNOWN",
                "file": source,
                "line": sec["line"]
            })
        elif name == "USER MATERIAL":
            values = [
                value
                for row in sec["data"]
                for value in numeric_tokens(row)
            ]
            user_materials.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "declared_constants": get_attribute(kw, "CONSTANTS"),
                "actual_numeric_values": len(values),
                "type": get_attribute(kw, "TYPE"),
                "values_preview": values[:12]
            })
        elif name == "DENSITY":
            density_blocks.append({
                "file": source,
                "line": sec["line"],
                "data": sec["data"]
            })
        elif name == "ORIENTATION":
            orientations.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
        elif name == "STEP":
            steps.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw
            })
        elif name == "BOUNDARY":
            boundary_conditions.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
        elif name == "AMPLITUDE":
            amplitudes.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data_preview": sec["data"][:8]
            })
        elif name in ("CLOAD", "DLOAD", "DSLOAD", "TEMPERATURE"):
            loads.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data_preview": sec["data"][:8]
            })
        elif name == "INITIAL CONDITIONS":
            initial_conditions.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data_preview": sec["data"][:8]
            })
        elif name in ("OUTPUT", "NODE OUTPUT", "ELEMENT OUTPUT"):
            output_requests.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
        elif name in ("SOLID SECTION", "SHELL SECTION"):
            sections_info.append({
                "file": source,
                "line": sec["line"],
                "keyword": kw,
                "data": sec["data"]
            })
    return {
        "main_deck": str(main_file.relative_to(ROOT)),
        "included_files": file_records,
        "included_file_count": len(file_records),
        "missing_includes": missing,
        "parse_errors": errors,
        "keyword_counts": dict(counts),
        "node_rows": len(node_rows),
        "element_rows_approx": len(element_rows),
        "element_types_approx": dict(element_types),
        "material_count": len(materials),
        "materials": materials,
        "user_materials": user_materials,
        "density_blocks": density_blocks,
        "orientation_blocks": orientations,
        "steps": steps,
        "boundary_conditions": boundary_conditions,
        "amplitudes": amplitudes,
        "loads": loads,
        "initial_conditions": initial_conditions,
        "output_requests": output_requests,
        "element_sections": sections_info,
        "node_row_examples": node_rows[:3],
        "element_row_examples": element_rows[:3]
    }
main_decks = sorted(
    p for p in DATASET.rglob("*.inp")
    if p.name.lower().startswith("job_")
)
reports = [audit_model(p) for p in main_decks]
# Save full recursive report.
(OUT / "abaqus_recursive_model_audit.json").write_text(
    json.dumps(reports, indent=2),
    encoding="utf-8"
)
# Save comparison table.
comparison = []
for r in reports:
    comparison.append({
        "main_deck": r["main_deck"],
        "included_files": r["included_file_count"],
        "missing_includes": len(r["missing_includes"]),
        "nodes": r["node_rows"],
        "element_rows_approx": r["element_rows_approx"],
        "element_types_approx": json.dumps(r["element_types_approx"]),
        "materials": r["material_count"],
        "user_material_blocks": len(r["user_materials"]),
        "orientations": len(r["orientation_blocks"]),
        "steps": len(r["steps"]),
        "boundary_blocks": len(r["boundary_conditions"]),
        "amplitude_blocks": len(r["amplitudes"]),
        "output_blocks": len(r["output_requests"]),
        "parse_errors": len(r["parse_errors"])
    })
with (OUT / "abaqus_recursive_mesh_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
    writer.writeheader()
    writer.writerows(comparison)
# Human-readable report.
lines = [
    "MAT-X RECURSIVE ABAQUS MODEL AUDIT",
    "=" * 90,
    f"Main job decks: {len(reports)}",
    "Counts aggregate recursively included input files.",
    "Element-row counts are approximate pending element-connectivity validation.",
    ""
]
for r in reports:
    lines += [
        "-" * 90,
        r["main_deck"],
        f"Included files: {r['included_file_count']}",
        f"Nodes: {r['node_rows']}",
        f"Approximate element rows: {r['element_rows_approx']}",
        f"Element types: {r['element_types_approx']}",
        f"Materials: {r['material_count']}",
        f"User material blocks: {len(r['user_materials'])}",
        f"Orientation blocks: {len(r['orientation_blocks'])}",
        f"Steps: {r['steps']}",
        f"Missing includes: {r['missing_includes']}",
        f"Parse errors: {r['parse_errors']}"
    ]
    for um in r["user_materials"]:
        lines.append(
            "USER MATERIAL: "
            f"{um['file']} line {um['line']} | "
            f"declared constants={um['declared_constants']} | "
            f"parsed numeric values={um['actual_numeric_values']} | "
            f"type={um['type']}"
        )
    for orient in r["orientation_blocks"]:
        lines.append(
            f"ORIENTATION: {orient['file']} line {orient['line']} "
            f"| {orient['keyword']}"
        )
    for amp in r["amplitudes"]:
        lines.append(
            f"AMPLITUDE: {amp['file']} line {amp['line']} "
            f"| {amp['keyword']}"
        )
lines += [
    "",
    "SCIENTIFIC LIMITATIONS",
    "=" * 90,
    "These are computational Abaqus input decks, not experimental measurements.",
    "No simulations were executed.",
    "Element counts are approximate because continuation-line handling varies by element type.",
    "Material constants are not interpreted without the constitutive model documentation.",
    "Units and physical meaning of state variables are not assumed."
]
(OUT / "abaqus_recursive_model_audit.txt").write_text(
    "\n".join(lines),
    encoding="utf-8"
)
print("\n" + "=" * 85)
print("RECURSIVE ABAQUS AUDIT COMPLETE")
print("=" * 85)
for row in comparison:
    print(row)
print("\nOUTPUT:", OUT)
print(" - abaqus_recursive_model_audit.json")
print(" - abaqus_recursive_mesh_comparison.csv")
print(" - abaqus_recursive_model_audit.txt")
