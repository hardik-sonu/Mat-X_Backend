import ast
from pathlib import Path
FILES = [
    ("app/engines/active_investigation_engine.py",
     ["recommend_next_investigation"]),
    ("app/engines/material_identity_engine.py",
     ["identify_material_candidates"]),
    ("app/engines/process_history_engine.py",
     ["reconstruct_process_history"]),
    ("app/services/microstructure_service.py",
     ["get_microstructure_features",
      "get_unified_microstructure_record"]),
    ("app/services/material_identity_service.py",
     ["identify_material"]),
    ("app/services/process_history_service.py",
     ["get_heat_process_history",
      "get_grade_process_history"]),
]
for filename, wanted in FILES:
    path = Path(filename)
    print("\n" + "=" * 85)
    print(filename)
    print("=" * 85)
    if not path.exists():
        print("NOT FOUND")
        continue
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted:
            continue
        print(f"\nFUNCTION: {node.name}")
        print("SIGNATURE:", ast.unparse(node.args))
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                print(
                    "RETURN at line",
                    child.lineno,
                    ":",
                    ast.unparse(child.value) if child.value else "None"
                )
        start = node.lineno
        end = node.end_lineno
        print("\nIMPLEMENTATION:")
        print("\n".join(
            f"{i}: {lines[i-1]}"
            for i in range(start, end + 1)
        ))
print("\n=== CONTRACT EXTRACTION COMPLETE ===")
