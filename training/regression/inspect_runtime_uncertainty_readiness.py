from pathlib import Path
import json
import pandas as pd
ROOT = Path("training_results")
keywords = ("applicability", "domain", "uncertainty", "calibration")
print("=" * 100)
print("MAT-X UNCERTAINTY / APPLICABILITY ARTIFACT INVENTORY")
print("=" * 100)
files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file() and any(k in p.name.lower() for k in keywords)
)
print(f"\nMatching files: {len(files)}")
for path in files:
    print(f"\n{path}  [{path.stat().st_size:,} bytes]")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                print("  Top-level keys:", list(data.keys())[:35])
                for key in (
                    "production_status", "validation_scope",
                    "external_validation", "status",
                    "n_records", "n_features", "n_folds"
                ):
                    if key in data:
                        print(f"  {key}: {data[key]}")
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=5)
            print("  Columns:", list(frame.columns))
            print("  Sample rows:", len(frame))
    except Exception as exc:
        print("  Inspection warning:", repr(exc))
print("\n" + "=" * 100)
print("CURRENT PREDICTION / APPLICABILITY SERVICE FILES")
print("=" * 100)
for path in [
    Path("app/services/prediction_service.py"),
    Path("app/core/model_registry.py"),
]:
    print(f"\n{path}: exists={path.exists()}")
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if any(k in line.lower() for k in (
                "def ", "class ", "applicability",
                "uncertainty", "input_evidence",
                "model_path", "joblib.load"
            )):
                print(f"{i}: {line.strip()}")
print("\nREAD-ONLY INVENTORY COMPLETE.")
