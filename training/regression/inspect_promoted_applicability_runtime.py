from pathlib import Path
import joblib
root = Path(r"C:\Users\Ahmadcomputer\MAT-X")
models = [
    "steelbench_tensile_extratrees.joblib",
    "steelbench_yield_extratrees.joblib",
]
for filename in models:
    path = root / "models" / filename
    model = joblib.load(path)
    print("\nMODEL:", filename)
    print("TYPE:", type(model).__name__)
    print("STEPS:", getattr(model, "named_steps", {}).keys())
    if hasattr(model, "feature_names_in_"):
        print("FEATURES:", list(model.feature_names_in_))
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            print("STEP:", name, type(step).__name__)
            if hasattr(step, "feature_names_in_"):
                print("STEP FEATURES:", list(step.feature_names_in_))
            if hasattr(step, "transformers_"):
                for transformer_name, transformer, columns in step.transformers_:
                    print(
                        "TRANSFORMER:",
                        transformer_name,
                        "COLUMNS:",
                        list(columns)
                    )
