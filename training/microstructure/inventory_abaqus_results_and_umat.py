from pathlib import Path
import json
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
result_extensions = {
    ".odb", ".dat", ".msg", ".sta", ".sim", ".fil",
    ".prt", ".com", ".log", ".res", ".sel", ".pac",
    ".stt", ".mdl", ".abq"
}
source_extensions = {
    ".for", ".f", ".f90", ".f95", ".f03", ".c",
    ".cpp", ".h", ".inc", ".py"
}
results = []
sources = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    ext = path.suffix.lower()
    record = {
        "relative_path": str(path.relative_to(root)),
        "extension": ext,
        "size_bytes": path.stat().st_size
    }
    if ext in result_extensions:
        results.append(record)
    if ext in source_extensions:
        sources.append(record)
print("=" * 90)
print("ABAQUS SIMULATION RESULT INVENTORY")
print("=" * 90)
print("Files found:", len(results))
for item in sorted(results, key=lambda x: x["relative_path"]):
    print(
        f'{item["extension"]:8} '
        f'{item["size_bytes"]:12,} bytes  '
        f'{item["relative_path"]}'
    )
print("\n" + "=" * 90)
print("CONSTITUTIVE / USER SUBROUTINE SOURCE INVENTORY")
print("=" * 90)
print("Files found:", len(sources))
for item in sorted(sources, key=lambda x: x["relative_path"]):
    print(
        f'{item["extension"]:8} '
        f'{item["size_bytes"]:12,} bytes  '
        f'{item["relative_path"]}'
    )
output = (
    Path(r"C:\Users\Ahmadcomputer\MAT-X")
    / "training_results"
    / "microstructure"
    / "abaqus_deep_audit"
)
output.mkdir(parents=True, exist_ok=True)
report = {
    "dataset_root": str(root),
    "simulation_result_count": len(results),
    "simulation_results": results,
    "constitutive_source_count": len(sources),
    "constitutive_sources": sources
}
report_path = output / "abaqus_results_and_umat_inventory.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("\nJSON REPORT:", report_path)
print("\nINVENTORY COMPLETE. ORIGINAL DATASET NOT MODIFIED.")
