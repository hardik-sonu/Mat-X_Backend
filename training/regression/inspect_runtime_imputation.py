from app.core.model_registry import MODELS
for model_name in ["tensile_strength", "yield_strength"]:
    model = MODELS[model_name]
    preprocessing = model.named_steps["preprocessing"]
    print("\nMODEL:", model_name)
    for transformer_name, transformer, columns in preprocessing.transformers_:
        print("\nTRANSFORMER:", transformer_name)
        print("COLUMNS:", list(columns))
        if transformer == "drop" or transformer == "passthrough":
            continue
        pipeline_steps = getattr(transformer, "named_steps", {})
        for step_name, step in pipeline_steps.items():
            print("STEP:", step_name, type(step).__name__)
            if hasattr(step, "statistics_"):
                print("IMPUTER STRATEGY:", step.strategy)
                print("IMPUTER STATISTICS:")
                for column, value in zip(columns, step.statistics_):
                    print(f"  {column}: {value!r}")
