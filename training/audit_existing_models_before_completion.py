from pathlib import Path
import json
import joblib
ROOT = Path.cwd()
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "training_results" / "regression"
model_files = sorted(MODEL_DIR.glob("*.joblib"))
print("=" * 80)
print("MAT-X EXISTING MODEL ARTIFACT AUDIT")
print("=" * 80)
for path in model_files:
    print(f"\nMODEL: {path.name}")
    try:
        obj = joblib.load(path)
        print("Type:", type(obj).__name__)
        if hasattr(obj, "named_steps"):
            print("Pipeline steps:", list(obj.named_steps.keys()))
            for name, step in obj.named_steps.items():
                print(f"  {name}: {type(step).__name__}")
        if hasattr(obj, "feature_names_in_"):
            print("Feature names:", list(obj.feature_names_in_))
        if hasattr(obj, "n_features_in_"):
            print("Input feature count:", obj.n_features_in_)
        if hasattr(obj, "estimators_"):
            print("Estimator count:", len(obj.estimators_))
        if hasattr(obj, "classes_"):
            print("Classes:", obj.classes_.tolist())
        if hasattr(obj, "n_outputs_"):
            print("Output count:", obj.n_outputs_)
    except Exception as exc:
        print("LOAD ERROR:", repr(exc))
print("\n" + "=" * 80)
print("REGRESSION TRAINING REPORTS")
print("=" * 80)
for path in sorted(REPORT_DIR.glob("*.json")):
    print(f"\nREPORT: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in [
            "target",
            "model",
            "algorithm",
            "validation_strategy",
            "training_records",
            "test_records",
            "mae",
            "rmse",
            "r2",
            "features",
            "limitations"
        ]:
            if key in data:
                print(f"{key}: {data[key]}")
    except Exception as exc:
        print("REPORT ERROR:", repr(exc))
print("\nAUDIT COMPLETE")
