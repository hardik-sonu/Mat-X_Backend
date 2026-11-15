from pathlib import Path
from collections import Counter
import json
import csv
import re
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\simulation")
OUT.mkdir(parents=True, exist_ok=True)
KEYWORDS = {
    "MATERIAL", "ELASTIC", "PLASTIC", "DENSITY", "USER MATERIAL",
    "SOLID SECTION", "SHELL SECTION", "BEAM SECTION",
    "ELEMENT", "NODE", "NSET", "ELSET", "ASSEMBLY", "INSTANCE",
    "STEP", "STATIC", "DYNAMIC", "FREQUENCY", "BOUNDARY",
    "CLOAD", "DLOAD", "DSLOAD", "AMPLITUDE", "OUTPUT",
    "NODE OUTPUT", "ELEMENT OUTPUT", "INCLUDE",
    "INITIAL CONDITIONS", "CONTACT", "SURFACE", "TIE",
    "EQUATION", "ORIENTATION", "DEPVAR", "EXPANSION",
    "CONDUCTIVITY", "SPECIFIC HEAT", "DAMAGE"
}
IMPORTANT = {
    "MATERIAL", "ELASTIC", "PLASTIC", "DENSITY", "USER MATERIAL",
    "SOLID SECTION", "ELEMENT", "NODE", "STEP", "STATIC",
    "DYNAMIC", "FREQUENCY", "BOUNDARY", "CLOAD", "DLOAD",
    "AMPLITUDE", "INCLUDE", "INITIAL CONDITIONS", "CONTACT",
    "SURFACE", "TIE", "ORIENTATION", "DEPVAR"
}
NUMBERED_ROW = re.compile(r"^\s*\d+\s*,")
MATERIAL_NAME = re.compile(r"\bNAME\s*=\s*([^,\s]+)", re.I)
INCLUDE_NAME = re.compile(r"\bINPUT\s*=\s*([^,\s]+)", re.I)
ELEMENT_TYPE = re.compile(r"\bTYPE\s*=\s*([^,\s]+)", re.I)
files = sorted(ROOT.rglob("*.inp"))
reports = []
print("Abaqus files discovered:", len(files))
for file_number, path in enumerate(files, start=1):
    print(f"[{file_number}/{len(files)}] Auditing: {path.name}")
    report = {
        "file": str(path),
        "relative_path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "line_count": 0,
        "node_records": 0,
        "element_records": 0,
        "element_types": set(),
        "materials": [],
        "steps": [],
        "include_references": [],
        "keyword_counts": Counter(),
        "keyword_details": [],
        "error": None
    }
    section = None
    current_material = None
    try:
        with path.open("r", encoding="latin-1", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                report["line_count"] = line_number
                line = raw.strip()
                if not line or line.startswith("**"):
                    continue
                if line.startswith("*"):
                    upper = line.upper()
                    keyword = upper[1:].split(",", 1)[0].strip()
                    if keyword in KEYWORDS:
                        report["keyword_counts"][keyword] += 1
                    if keyword in IMPORTANT and len(report["keyword_details"]) < 10000:
                        report["keyword_details"].append({
                            "line": line_number,
                            "keyword": keyword,
                            "text": line[:300]
                        })
                    section = None
                    if keyword == "NODE":
                        section = "NODE"
                    elif keyword == "ELEMENT":
                        section = "ELEMENT"
                        match = ELEMENT_TYPE.search(line)
                        if match:
                            report["element_types"].add(match.group(1))
                    elif keyword == "MATERIAL":
                        match = MATERIAL_NAME.search(line)
                        current_material = {
                            "line": line_number,
                            "name": match.group(1) if match else None,
                            "definitions": []
                        }
                        report["materials"].append(current_material)
                    elif keyword == "STEP":
                        report["steps"].append({
                            "line": line_number,
                            "definition": line[:300]
                        })
                    elif keyword == "INCLUDE":
                        match = INCLUDE_NAME.search(line)
                        report["include_references"].append(
                            match.group(1) if match else line
                        )
                    if current_material is not None and keyword in {
                        "ELASTIC", "PLASTIC", "DENSITY", "USER MATERIAL",
                        "DEPVAR", "EXPANSION", "CONDUCTIVITY",
                        "SPECIFIC HEAT", "DAMAGE"
                    }:
                        current_material["definitions"].append(line[:300])
                    continue
                if section == "NODE" and NUMBERED_ROW.match(line):
                    report["node_records"] += 1
                elif section == "ELEMENT" and NUMBERED_ROW.match(line):
                    report["element_records"] += 1
                elif current_material is not None:
                    # Preserve material-property data rows.
                    if len(current_material["definitions"]) < 100:
                        current_material["definitions"].append(line[:300])
        report["element_types"] = sorted(report["element_types"])
        report["keyword_counts"] = dict(report["keyword_counts"])
    except Exception as exc:
        report["error"] = repr(exc)
    reports.append(report)
json_path = OUT / "abaqus_input_audit.json"
csv_path = OUT / "abaqus_input_audit_summary.csv"
txt_path = OUT / "abaqus_input_audit_readable.txt"
json_path.write_text(
    json.dumps(reports, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
summary = []
for r in reports:
    summary.append({
        "relative_path": r["relative_path"],
        "size_bytes": r["size_bytes"],
        "line_count": r["line_count"],
        "node_records": r["node_records"],
        "element_records": r["element_records"],
        "element_types": ";".join(r["element_types"]),
        "material_count": len(r["materials"]),
        "material_names": ";".join(
            str(m["name"]) for m in r["materials"]
        ),
        "step_count": len(r["steps"]),
        "step_definitions": ";".join(
            s["definition"] for s in r["steps"]
        ),
        "include_references": ";".join(r["include_references"]),
        "error": r["error"] or ""
    })
if summary:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
with txt_path.open("w", encoding="utf-8") as handle:
    handle.write("MAT-X ABAQUS INPUT FILE AUDIT\n")
    handle.write("=" * 100 + "\n")
    handle.write(f"Files discovered: {len(files)}\n")
    for r in reports:
        handle.write("\n" + "=" * 100 + "\n")
        handle.write(f"FILE: {r['relative_path']}\n")
        if r["error"]:
            handle.write(f"ERROR: {r['error']}\n")
            continue
        handle.write(f"Size: {r['size_bytes']} bytes\n")
        handle.write(f"Lines: {r['line_count']}\n")
        handle.write(f"Nodes: {r['node_records']}\n")
        handle.write(f"Elements: {r['element_records']}\n")
        handle.write(f"Element types: {r['element_types']}\n")
        handle.write(f"Materials: {json.dumps(r['materials'])}\n")
        handle.write(f"Steps: {json.dumps(r['steps'])}\n")
        handle.write(f"Includes: {r['include_references']}\n")
        handle.write(f"Keyword counts: {r['keyword_counts']}\n")
        for item in r["keyword_details"]:
            handle.write(
                f"  L{item['line']}: {item['text']}\n"
            )
print("\n========== ABAQUS AUDIT COMPLETE ==========")
print("Files discovered:", len(files))
print("Files with errors:", sum(bool(r["error"]) for r in reports))
print("JSON:", json_path)
print("CSV :", csv_path)
print("TXT :", txt_path)
print("\n========== FILE SUMMARY ==========")
for r in reports:
    if r["error"]:
        print("ERROR:", r["relative_path"], r["error"])
    else:
        print(
            f"{r['relative_path']} | "
            f"lines={r['line_count']} | "
            f"nodes={r['node_records']} | "
            f"elements={r['element_records']} | "
            f"materials={len(r['materials'])} | "
            f"steps={len(r['steps'])} | "
            f"types={r['element_types']}"
        )
