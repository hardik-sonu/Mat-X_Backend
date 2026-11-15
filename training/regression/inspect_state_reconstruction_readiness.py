from pathlib import Path
import ast
import json
ROOT = Path(".")
APP = ROOT / "app"
TESTS = ROOT / "tests"
keywords = (
    "state", "uncertainty", "applicability", "investigation",
    "material", "evidence", "provenance", "report", "microstructure"
)
print("=== 1. RELEVANT BACKEND FILES ===")
for folder in [APP, TESTS]:
    if not folder.exists():
        print(f"\n{folder}: NOT FOUND")
        continue
    print(f"\n{folder}:")
    for path in sorted(folder.rglob("*.py")):
        if any(word in str(path).lower() for word in keywords):
            print(path)
print("\n=== 2. API ROUTES ===")
routes = APP / "api" / "routes.py"
if routes.exists():
    tree = ast.parse(routes.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [
                ast.unparse(d) for d in node.decorator_list
            ]
            if any(
                any(method in d.lower() for method in
                    ["get(", "post(", "put(", "delete(", "patch("])
                for d in decorators
            ):
                print(f"{node.name}: {' | '.join(decorators)}")
print("\n=== 3. ENGINE / SERVICE CLASS AND FUNCTION INVENTORY ===")
for folder in [APP / "engines", APP / "services"]:
    if not folder.exists():
        continue
    for path in sorted(folder.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            print(f"PARSE ERROR {path}: {exc}")
            continue
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                symbols.append(
                    f"{node.__class__.__name__}: {node.name}"
                )
        if symbols:
            print(f"\n{path}")
            for symbol in symbols:
                print("  ", symbol)
print("\n=== 4. EXISTING STATE / UNCERTAINTY REFERENCES ===")
for folder in [APP, TESTS]:
    if not folder.exists():
        continue
    for path in sorted(folder.rglob("*.py")):
        try:
            lines = path.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines()
        except Exception:
            continue
        matches = []
        for number, line in enumerate(lines, 1):
            if any(word in line.lower() for word in (
                "material_state", "state_reconstruction",
                "uncertainty", "applicability",
                "evidence_completeness", "next_measurement",
                "information_gain", "provenance"
            )):
                matches.append((number, line.strip()))
        if matches:
            print(f"\n{path}")
            for number, line in matches[:15]:
                print(f"  L{number}: {line[:180]}")
print("\n=== 5. TEST FILE INVENTORY ===")
if TESTS.exists():
    for path in sorted(TESTS.rglob("test_*.py")):
        print(path)
print("\n=== 6. INVENTORY COMPLETE ===")
print("No source files or model artifacts were modified.")
