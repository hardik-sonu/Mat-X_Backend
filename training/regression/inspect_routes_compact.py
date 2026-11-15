from pathlib import Path
import ast
path = Path("app/api/routes.py")
source = path.read_text(encoding="utf-8-sig")
lines = source.splitlines()
tree = ast.parse(source)
print("=== IMPORTS AND ROUTER SETUP ===")
for i, line in enumerate(lines, 1):
    if i > 45:
        break
    print(f"{i}: {line}")
print("\n=== EXISTING ROUTE HANDLERS ===")
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        decorators = [
            ast.unparse(d) for d in node.decorator_list
        ]
        if any("router." in d for d in decorators):
            print(f"\nFUNCTION: {node.name}")
            print("DECORATORS:", decorators)
            start = node.lineno
            end = min(node.end_lineno, start + 17)
            for i in range(start, end + 1):
                print(f"{i}: {lines[i-1]}")
print("\n=== ROUTE INSPECTION COMPLETE ===")
