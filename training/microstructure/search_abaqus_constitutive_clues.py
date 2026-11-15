from pathlib import Path
import re
import json
roots = [
    Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\Grain orientation and cyclic behavior"),
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
]
extensions = {
    ".inp", ".for", ".f", ".f90", ".f95", ".f03", ".inc",
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml",
    ".py", ".dat", ".log", ".xml", ".html", ".tex", ".bib"
}
patterns = {
    "umat": re.compile(r"\bUMAT\b", re.I),
    "vumat": re.compile(r"\bVUMAT\b", re.I),
    "subroutine": re.compile(r"\bSUBROUTINE\b", re.I),
    "crystal_plasticity": re.compile(r"crystal\s+plasticity|crystal\s+plastic", re.I),
    "orientation": re.compile(r"orientation|Euler|rotation matrix", re.I),
    "constitutive": re.compile(r"constitutive|user material", re.I),
    "slip_system": re.compile(r"slip system|Schmid|resolved shear", re.I),
    "hardening": re.compile(r"hardening|dislocation density", re.I),
    "source_reference": re.compile(r"doi.org|github.com|supplementary|supporting information", re.I),
}
filename_terms = (
    "umat", "vumat", "crystal", "plastic", "orientation",
    "material", "supplement", "readme", "source", "subroutine"
)
file_hits = []
text_hits = []
errors = []
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.lower() in {".venv", "node_modules", ".git", "__pycache__"}
               for part in path.parts):
            continue
        if any(term in path.name.lower() for term in filename_terms):
            file_hits.append(str(path))
        if path.suffix.lower() not in extensions:
            continue
        try:
            if path.stat().st_size > 15_000_000:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(content.splitlines(), start=1):
                matched = [name for name, pattern in patterns.items()
                           if pattern.search(line)]
                if matched:
                    text_hits.append({
                        "file": str(path),
                        "line": line_number,
                        "matches": matched,
                        "text": line.strip()[:350]
                    })
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
output = Path("training_results/microstructure/abaqus_deep_audit")
output.mkdir(parents=True, exist_ok=True)
(output / "constitutive_clue_search.json").write_text(
    json.dumps({
        "filename_hits": file_hits,
        "text_hits": text_hits,
        "errors": errors
    }, indent=2),
    encoding="utf-8"
)
print("FILENAME MATCHES:", len(file_hits))
for item in file_hits[:100]:
    print("FILE:", item)
print("\nTEXT MATCHES:", len(text_hits))
for item in text_hits[:160]:
    print(
        f'{item["file"]} | line {item["line"]} | '
        f'{",".join(item["matches"])} | {item["text"]}'
    )
print("\nREAD ERRORS:", len(errors))
print("FULL REPORT:", output / "constitutive_clue_search.json")
