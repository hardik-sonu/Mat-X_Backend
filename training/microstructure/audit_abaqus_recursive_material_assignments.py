from pathlib import Path
import re
import json
from collections import Counter
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
keyword_pattern = re.compile(r"^\s*\*([A-Z][A-Z0-9 _-]*)", re.I)
include_pattern = re.compile(
    r"^\s*\*INCLUDE\s*,\s*INPUT\s*=\s*(.+?)\s*$", re.I
)
interesting = {
    "MATERIAL", "USER MATERIAL", "DEPVAR",
    "SOLID SECTION", "ORIENTATION", "DISTRIBUTION",
    "ELSET", "ELEMENT", "INCLUDE", "INITIAL CONDITIONS"
}
def resolve_file(parent, name):
    name = name.strip().strip('"').strip("'")
    for candidate in [(parent / name).resolve(), (root / name).resolve()]:
        if candidate.exists():
            return candidate
    return None
def collect_recursive(path, visited=None):
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited or not path.exists():
        return [], []
    visited.add(path)
    try:
        lines = path.read_text(
            encoding="utf-8-sig", errors="replace"
        ).splitlines()
    except Exception:
        return [], []
    records, missing = [], []
    for number, line in enumerate(lines, 1):
        match = keyword_pattern.match(line)
        if match:
            keyword = " ".join(match.group(1).upper().split())
            base = keyword.split(",")[0].strip()
            if base in interesting:
                records.append({
                    "file": str(path.relative_to(root)),
                    "line": number,
                    "keyword": base,
                    "text": line.strip()
                })
        include = include_pattern.match(line)
        if include:
            included = resolve_file(path.parent, include.group(1))
            if included is None:
                missing.append({
                    "file": str(path.relative_to(root)),
                    "line": number,
                    "include": include.group(1).strip()
                })
            else:
                child_records, child_missing = collect_recursive(
                    included, visited
                )
                records.extend(child_records)
                missing.extend(child_missing)
    return records, missing
report = {"models": [], "summary": {}}
for job in sorted(root.glob("Mesh*/Job_*.inp")):
    records, missing = collect_recursive(job)
    counts = Counter(x["keyword"] for x in records)
    model = {
        "job": str(job.relative_to(root)),
        "included_file_count": len({x["file"] for x in records}),
        "keyword_counts": dict(counts),
        "missing_includes": missing,
        "records": records
    }
    report["models"].append(model)
    print("\n" + "=" * 85)
    print("MODEL:", job.parent.name)
    print("=" * 85)
    for keyword in sorted(interesting):
        print(f"{keyword:24}: {counts.get(keyword, 0)}")
    print("Missing includes:", len(missing))
    for item in records:
        if item["keyword"] in {
            "DEPVAR", "ORIENTATION", "DISTRIBUTION",
            "SOLID SECTION", "USER MATERIAL", "INCLUDE"
        }:
            print(
                f'  {item["keyword"]:18} '
                f'{item["file"]}:{item["line"]} '
                f'{item["text"]}'
            )
report["summary"] = {
    "job_count": len(report["models"]),
    "models_with_missing_includes": sum(
        bool(x["missing_includes"]) for x in report["models"]
    )
}
report_path = output / "abaqus_recursive_material_assignment_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nSUMMARY:", json.dumps(report["summary"], indent=2))
print("REPORT:", report_path)
print("Original input files were not modified.")
