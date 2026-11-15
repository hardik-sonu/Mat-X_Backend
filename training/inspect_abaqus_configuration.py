from pathlib import Path
import json
report_path = Path(
    r"training_results\simulation\abaqus_input_audit.json"
)
reports = json.loads(report_path.read_text(encoding="utf-8"))
print("\n" + "=" * 100)
print("MAT-X ABAQUS MODEL CONFIGURATION INSPECTION")
print("=" * 100)
for r in reports:
    name = Path(r["relative_path"]).name.lower()
    if not (
        name.startswith("job_")
        or name.endswith("_mat.inp")
        or name in {"amplitude.inp", "timepoints.inp"}
    ):
        continue
    print("\n" + "-" * 100)
    print("FILE:", r["relative_path"])
    print("LINES:", r["line_count"])
    print("\nKEYWORD COUNTS:")
    for key, value in sorted(r["keyword_counts"].items()):
        print(f"  {key}: {value}")
    print("\nMATERIAL DEFINITIONS:")
    for material in r["materials"]:
        print(
            f"\n  Material: {material.get('name')} "
            f"(line {material.get('line')})"
        )
        for definition in material.get("definitions", []):
            print("   ", definition)
    print("\nSIMULATION STEPS:")
    for step in r["steps"]:
        print("  ", step)
    print("\nINCLUDE REFERENCES:")
    for include in r["include_references"]:
        print("  ", include)
    print("\nIMPORTANT KEYWORD LINES:")
    for item in r["keyword_details"]:
        print(
            f"  L{item['line']} "
            f"[{item['keyword']}] {item['text']}"
        )
print("\n" + "=" * 100)
print("CONFIGURATION INSPECTION COMPLETE")
print("=" * 100)
