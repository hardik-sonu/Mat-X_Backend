from pathlib import Path
from collections import Counter, defaultdict
import csv
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure\abaqus_deep_audit")
OUT.mkdir(parents=True, exist_ok=True)
result_extensions = {
    ".odb", ".dat", ".msg", ".sta", ".sim",
    ".fil", ".prt", ".com", ".log", ".res",
    ".sel", ".pac", ".stt", ".mdl", ".abq"
}
source_extensions = {
    ".for", ".f", ".f90", ".f95", ".c", ".cpp",
    ".h", ".inc", ".py"
}
extension_counts = Counter()
result_files = []
source_files = []
other_files = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    ext = path.suffix.lower()
    extension_counts[ext or "[no extension]"] += 1
    record = {
        "relative_path": str(path.relative_to(ROOT)),
        "extension": ext,
        "size_bytes": path.stat().st_size
    }
    if ext in result_extensions:
        result_files.append(record)
    elif ext in source_extensions:
        source_files.append(record)
# Identify likely UMAT / constitutive implementation files by filename.
likely_umat = [
    x for x in source_files
    if any(term in Path(x["relative_path"]).name.lower()
           for term in ("umat", "material", "crystal", "plastic", "hardening"))
]
def save_csv(filename, records):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        fields = ["relative_path", "extension", "size_bytes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
save_csv("abaqus_result_file_inventory.csv", result_files)
save_csv("abaqus_source_file_inventory.csv", source_files)
save_csv("abaqus_likely_umat_files.csv", likely_umat)
print("\n" + "=" * 85)
print("ABAQUS RESULTS AND USER-MATERIAL SOURCE INVENTORY")
print("=" * 85)
print("\nTOTAL DATASET FILES:", sum(extension_counts.values()))
print("\nRESULT FILES:", len(result_files))
for x in result_files:
    print(f"{x['extension']:8} {x['size_bytes']:12,} bytes | {x['relative_path']}")
print("\nSOURCE / CODE FILES:", len(source_files))
for x in source_files:
    print(f"{x['extension']:8} {x['size_bytes']:12,} bytes | {x['relative_path']}")
print("\nLIKELY UMAT / CONSTITUTIVE FILES:", len(likely_umat))
for x in likely_umat:
    print(" ", x["relative_path"])
print("\nEXTENSION COUNTS:")
for ext, count in extension_counts.most_common():
    print(f"{ext:18} {count}")
print("\nOUTPUT DIRECTORY:", OUT)
print(" - abaqus_result_file_inventory.csv")
print(" - abaqus_source_file_inventory.csv")
print(" - abaqus_likely_umat_files.csv")
print("\nAUDIT COMPLETE")
