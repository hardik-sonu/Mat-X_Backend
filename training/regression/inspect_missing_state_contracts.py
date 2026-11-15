import ast
from pathlib import Path
targets = {
    "app/engines/active_investigation_engine.py":
        ["recommend_next_investigation"],
    "app/engines/material_identity_engine.py":
        ["identify_material_candidates"],
    "app/services/microstructure_service.py":
        ["get_microstructure_features",
         "get_unified_microstructure_record"],
}
for filename, names in targets.items():
    path = Path(filename)
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        print(f"\n{'=' * 80}\n{filename} :: {node.name}\n{'=' * 80}")
        print(f"Lines {node.lineno}-{node.end_lineno}")
        # Print function implementation, capped at 100 lines.
        end = min(node.end_lineno, node.lineno + 99)
        for i in range(node.lineno, end + 1):
            print(f"{i}: {lines[i-1]}")
print("\n=== REQUIRED COMPONENTS EXTRACTED ===")
