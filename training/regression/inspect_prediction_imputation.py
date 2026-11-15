import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
models = [
    "steelbench_tensile_extratrees.joblib",
    "steelbench_yield_extratrees.joblib",
]
for filename in models:
    print("\n" + "=" * 75)
    print("MODEL:", filename)
    artifact = joblib.load(Path("models") / filename)
    preprocessing = artifact.named_steps["preprocessing"]
    print("\nPreprocessing type:", type(preprocessing).__name__)
    for name, transformer, columns in preprocessing.transformers_:
        print("\nTransformer:", name)
        print("Columns:", columns)
        print("Type:", type(transformer).__name__)
        if hasattr(transformer, "steps"):
            for step_name, step in transformer.steps:
                print("  Step:", step_name, type(step).__name__)
                if hasattr(step, "statistics_"):
                    print("  Imputation statistics:", step.statistics_)
                if hasattr(step, "strategy"):
                    print("  Imputation strategy:", step.strategy)
        elif hasattr(transformer, "statistics_"):
            print("Imputation statistics:", transformer.statistics_)
        elif hasattr(transformer, "strategy"):
            print("Imputation strategy:", transformer.strategy)
    print("\nPipeline inspection complete.")
