import ast
from pathlib import Path
FILES = {
    "app/engines/active_investigation_engine.py": [
        "recommend_next_investigation",
    ],
    "app/engines/material_identity_engine.py": [
        "identify_material_candidates",
    ],
    "app/engines/process_history_engine.py": [
        "reconstruct_process_history",
    ],
    "app/services/microstructure_service.py": [
        "get_microstructure_features",
        "get_unified_microstructure_record",
    ],
    "app/services/prediction_service.py": [
        "predict_tensile",
        "predict_yield",
        "predict_inclusion",
    ],
}
for filename, selected in FILES.items():
    path = Path(filename)
    print("\n" + "=" * 90)
    print("FILE:", filename)
    print("=" * 90)
    if not path.exists():
        print("NOT FOUND")
        continue
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    tree = ast.parse(source)
    print("\n--- IMPORTS AND MODULE CONSTANTS ---")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign,
                             ast.AnnAssign)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            print("\n".join(
                f"{i}: {lines[i-1]}"
                for i in range(start, end + 1)
            ))
    print("\n--- SELECTED FUNCTION IMPLEMENTATIONS ---")
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in selected:
                found.add(node.name)
                start = node.lineno
                end = node.end_lineno
                print(f"\n### {node.name} (lines {start}-{end})")
                print("\n".join(
                    f"{i}: {lines[i-1]}"
                    for i in range(start, end + 1)
                ))
    for name in selected:
        if name not in found:
            print(f"\nFUNCTION NOT FOUND: {name}")
SCHEMAS = [
    "app/schemas/investigation.py",
    "app/schemas/material_identity.py",
    "app/schemas/predictions.py",
]
for filename in SCHEMAS:
    path = Path(filename)
    print("\n" + "=" * 90)
    print("SCHEMA:", filename)
    print("=" * 90)
    if path.exists():
        for number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            print(f"{number}: {line}")
    else:
        print("NOT FOUND")
print("\n=== COMPONENT CONTRACT INSPECTION COMPLETE ===")
