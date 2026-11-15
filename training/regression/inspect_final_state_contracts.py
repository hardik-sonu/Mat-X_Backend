import ast
from pathlib import Path
checks = {
    "app/engines/active_investigation_engine.py": [
        "recommend_next_investigation"
    ],
    "app/engines/material_identity_engine.py": [
        "identify_material_candidates"
    ],
}
for filename, wanted in checks.items():
    path = Path(filename)
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            print("\n" + "=" * 90)
            print(filename, node.name)
            print("=" * 90)
            for i in range(node.lineno, node.end_lineno + 1):
                print(f"{i}: {lines[i-1]}")
print("\n=== FINAL CONTRACT CHECK COMPLETE ===")
