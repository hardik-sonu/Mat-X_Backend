import json
from pathlib import Path
import pandas as pd
root = Path("training_results")
v1 = root / "applicability_domain_v1"
v2 = root / "applicability_domain_v2"
comparison = root / "applicability_domain_comparison_v1"
def inspect_csv(path, title, preferred=None):
    print(f"\n=== {title} ===")
    if not path.exists():
        print("NOT FOUND:", path)
        return
    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    if preferred:
        cols = [c for c in preferred if c in df.columns]
        if cols:
            print(df[cols].to_string(index=False))
    else:
        print(df.to_string(index=False, max_rows=60))
for version, folder in [("V1", v1), ("V2", v2)]:
    print(f"\n{'='*25} {version} SUMMARY {'='*25}")
    path = folder / "summary.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({
            "successful_assessments": data.get("successful_assessments"),
            "failed_assessments": data.get("failed_assessments"),
            "record_rows": data.get("record_rows"),
            "status_totals": data.get("status_totals"),
            "method": data.get("method"),
        }, indent=2))
    else:
        print("Summary not found:", path)
inspect_csv(
    comparison / "status_transition_matrix.csv",
    "V1 TO V2 STATUS TRANSITIONS"
)
inspect_csv(
    comparison / "source_status_comparison.csv",
    "SOURCE-LEVEL STATUS COMPARISON"
)
inspect_csv(
    comparison / "fold_comparison.csv",
    "FOLD-LEVEL COMPARISON"
)
print("\n=== V2 OUTPUT INTEGRITY ===")
for name in [
    "record_applicability.csv",
    "fold_applicability_summary.csv",
    "feature_domain_summary.csv",
    "split_integrity_audit.csv",
    "execution_failures.csv",
]:
    path = v2 / name
    if path.exists():
        df = pd.read_csv(path)
        print(f"{name}: {len(df)} rows")
        if name == "split_integrity_audit.csv":
            print(df.to_string(index=False))
        elif name == "execution_failures.csv":
            print(df.to_string(index=False))
    else:
        print(f"{name}: NOT FOUND")
