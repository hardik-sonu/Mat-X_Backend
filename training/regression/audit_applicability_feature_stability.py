import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATASET = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T", "quench_medium", "steel_family"
]
NUMERIC = FEATURES[:11]
CATEGORICAL = FEATURES[11:]
TARGETS = {
    "tensile_strength": {
        "model": ROOT / "models/steelbench_tensile_extratrees.joblib",
    },
    "yield_strength": {
        "model": ROOT / "models/steelbench_yield_extratrees.joblib",
    },
}
df = pd.read_csv(DATASET)
results = {}
for target, config in TARGETS.items():
    clean = df.dropna(subset=[target, "grade_id"]).copy()
    if target == "yield_strength":
        ratio = clean[target] / clean["tensile_strength"].replace(0, np.nan)
        anomaly = (
            clean["grade_id"].astype(str).isin(
                ["EMK0319", "EMK0320", "EMK0321"]
            )
            & (ratio < 0.03)
        )
        clean = clean.loc[~anomaly].copy()
    X = clean[FEATURES]
    groups = clean["grade_id"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, _ = next(splitter.split(X, groups=groups))
    train = X.iloc[train_idx].copy()
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    train = train.reset_index(drop=True)
    numeric_validity = {}
    for feature in NUMERIC:
        values = pd.to_numeric(train[feature], errors="coerce")
        numeric_validity[feature] = {
            "valid": int(values.notna().sum()),
            "unique": int(values.nunique()),
            "missing_fraction": float(values.isna().mean()),
        }
    categorical_validity = {}
    for feature in CATEGORICAL:
        values = train[feature].dropna().astype(str)
        categorical_validity[feature] = {
            "valid": int(len(values)),
            "unique": int(values.nunique()),
            "categories": sorted(values.unique().tolist()),
        }
    unstable = []
    for grade in sorted(train_groups.unique()):
        reference = train.loc[train_groups != grade]
        for feature in NUMERIC:
            values = pd.to_numeric(reference[feature], errors="coerce").dropna()
            if len(values) == 0 or values.nunique() <= 1:
                unstable.append({
                    "held_out_grade": grade,
                    "feature": feature,
                    "status": "ALL_MISSING_OR_CONSTANT",
                })
        for feature in CATEGORICAL:
            values = reference[feature].dropna().astype(str)
            if values.nunique() <= 1:
                unstable.append({
                    "held_out_grade": grade,
                    "feature": feature,
                    "status": "ALL_MISSING_OR_ONE_CATEGORY",
                })
    results[target] = {
        "training_records": len(train),
        "training_grades": int(train_groups.nunique()),
        "numeric_features": numeric_validity,
        "categorical_features": categorical_validity,
        "leave_one_grade_out_instability_count": len(unstable),
        "leave_one_grade_out_instability_examples": unstable[:30],
    }
    print(f"\n========== {target.upper()} ==========")
    print("Training records:", len(train))
    print("Training grades:", train_groups.nunique())
    print("LOO feature-instability events:", len(unstable))
    print("Numeric features:")
    print(json.dumps(numeric_validity, indent=2))
    print("Categorical features:")
    print(json.dumps(categorical_validity, indent=2))
    if unstable:
        print("Instability examples:")
        print(json.dumps(unstable[:15], indent=2))
out = ROOT / "training_results/applicability_model_alignment_audit"
out.mkdir(parents=True, exist_ok=True)
path = out / "leave_one_grade_out_feature_stability.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nReport:", path)
