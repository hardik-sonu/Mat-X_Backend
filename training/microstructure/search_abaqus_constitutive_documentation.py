from pathlib import Path
import re
import json
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
output = (
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
    / "training_results" / "microstructure"
    / "abaqus_deep_audit"
)
output.mkdir(parents=True, exist_ok=True)
keywords = [
    "UMAT", "VUMAT", "crystal plasticity",
    "crystallographic", "orientation", "direction cosine",
    "slip system", "slip systems", "hardening",
    "dislocation", "constitutive", "state variable",
    "depvar", "user material", "elastic constants",
    "single crystal", "316L", "Praxair"
]
extensions = {
    ".txt", ".md", ".rst", ".inp", ".for", ".f",
    ".f90", ".f95", ".inc", ".py", ".csv", ".dat"
}
results = []
material_files = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.name.lower() in {
        "hom_316l_mat.inp", "hom_316l_grains.inp"
    }:
        material_files.append(path)
    if path.suffix.lower() not in extensions:
        continue
    try:
        lines = path.read_text(
            encoding="utf-8-sig", errors="replace"
        ).splitlines()
    except Exception:
        continue
    for number, line in enumerate(lines, 1):
        lower = line.lower()
        matched = [k for k in keywords if k.lower() in lower]
        if matched:
            results.append({
                "file": str(path.relative_to(root)),
                "line": number,
                "keywords": matched,
                "text": line.strip()[:500]
            })
print("=" * 100)
print("MATERIAL MODEL DOCUMENTATION SEARCH")
print("=" * 100)
print("\nMaterial include files found:", len(material_files))
for path in material_files:
    print("\nFILE:", path.relative_to(root))
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    print("--- First 100 lines ---")
    for i, line in enumerate(lines[:100], 1):
        print(f"{i:5}: {line[:300]}")
print("\n" + "=" * 100)
print("KEYWORD MATCHES")
print("=" * 100)
for item in results[:500]:
    print(
        f'\n{item["file"]}:{item["line"]}'
        f'\nKeywords: {", ".join(item["keywords"])}'
        f'\n{item["text"]}'
    )
report_path = output / "abaqus_constitutive_documentation_search.json"
report_path.write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print("\nTotal keyword matches:", len(results))
print("Report:", report_path)
print("Original dataset files were not modified.")
