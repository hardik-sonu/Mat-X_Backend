import ast
from pathlib import Path
path = Path("training/regression/run_applicability_domain_assessment_v3.py")
# utf-8-sig safely removes a BOM if one exists
source = path.read_text(encoding="utf-8-sig")
lines = source.splitlines()
tree = ast.parse(source)
print("\n========== FUNCTION DEFINITIONS ==========")
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        print(f"{node.lineno:4d}-{node.end_lineno:4d}  {node.name}()")
keywords = (
    "RobustScaler", "StandardScaler", "NearestNeighbors",
    "pairwise_distances", "cdist", "quantile", "percentile",
    "distance", "calibration", "eligible_for_distance",
    "IN_DOMAIN", "OUT_OF_DOMAIN", "BORDERLINE",
    "INSUFFICIENT_EVIDENCE", "q95", "q99"
)
print("\n========== RELEVANT SOURCE LINES ==========")
for i, line in enumerate(lines, start=1):
    if any(word.lower() in line.lower() for word in keywords):
        print(f"{i:4d}: {line[:220]}")
print("\n========== IMPORTS / CONFIGURATION (FIRST 180 LINES) ==========")
for i, line in enumerate(lines[:180], start=1):
    print(f"{i:4d}: {line}")
