from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "applicability_feature_eligibility_v3"
RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.25
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
YIELD_ANOMALY_GRADES = ["EMK0319", "EMK0320", "EMK0321"]
BASE_NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si",
    "V", "Cu", "Al", "austenitize_T", "temper_T",
]
BASE_CATEGORICAL = ["quench_medium", "steel_family"]
VARIANTS = {
    "full_13": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
    },
    "chemistry_2": {
        "numeric": ["C", "Mn"],
        "categorical": [],
    },
    "chemistry_3": {
        "numeric": ["C", "Mn", "V"],
        "categorical": [],
    },
}
def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing directory: {OUT}"
        )
    OUT.mkdir(parents=True)
    df = pd.read_csv(DATASET)
    for column in BASE_NUMERIC + TARGETS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    rows = []
    for target in TARGETS:
        clean = df.dropna(
            subset=[target, "source", "grade_id", "heat_id"]
        ).copy()
        if target == "yield_strength":
            ratio = (
                clean[target]
                / clean["tensile_strength"].replace(0, np.nan)
            )
            anomaly = (
                clean["grade_id"].astype(str).isin(
                    YIELD_ANOMALY_GRADES
                )
                & (ratio < 0.03)
            )
            clean = clean.loc[~anomaly].copy()
        clean["grade_id"] = clean["grade_id"].astype(str)
        for held_out in SOURCES:
            test = clean.loc[
                clean["source"] == held_out
            ].copy()
            train_cal = clean.loc[
                clean["source"] != held_out
            ].copy()
            if test.empty or train_cal.empty:
                continue
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=CALIBRATION_FRACTION,
                random_state=(
                    RANDOM_STATE + SOURCES.index(held_out)
                ),
            )
            train_idx, calibration_idx = next(
                splitter.split(
                    np.zeros((len(train_cal), 1)),
                    train_cal[target].to_numpy(dtype=float),
                    groups=train_cal["grade_id"].to_numpy(),
                )
            )
            train = train_cal.iloc[train_idx].copy()
            for variant, config in VARIANTS.items():
                features = (
                    config["numeric"] + config["categorical"]
                )
                for feature in features:
                    is_numeric = feature in config["numeric"]
                    train_values = train[feature]
                    test_values = test[feature]
                    if is_numeric:
                        train_values = pd.to_numeric(
                            train_values, errors="coerce"
                        )
                        test_values = pd.to_numeric(
                            test_values, errors="coerce"
                        )
                    train_valid = train_values.dropna()
                    test_valid = test_values.dropna()
                    unique_train = train_valid.nunique()
                    unique_test = test_valid.nunique()
                    if len(train_valid) == 0:
                        status = "ALL_MISSING_IN_TRAIN"
                    elif is_numeric and unique_train == 1:
                        status = "CONSTANT_IN_TRAIN"
                    elif unique_train == 1:
                        status = "ONE_CATEGORY_IN_TRAIN"
                    else:
                        status = "VARIABLE_IN_TRAIN"
                    unsupported_count = 0
                    unseen_count = 0
                    constant_deviation_count = 0
                    if status == "CONSTANT_IN_TRAIN":
                        reference = float(train_valid.iloc[0])
                        constant_deviation_count = int(
                            (
                                test_valid.astype(float)
                                != reference
                            ).sum()
                        )
                        unsupported_count = (
                            constant_deviation_count
                        )
                    elif status == "ALL_MISSING_IN_TRAIN":
                        unsupported_count = len(test_valid)
                    elif not is_numeric:
                        known = set(train_valid.astype(str))
                        unseen = (
                            ~test_valid.astype(str).isin(known)
                        )
                        unseen_count = int(unseen.sum())
                        unsupported_count = unseen_count
                    rows.append({
                        "target": target,
                        "held_out_source": held_out,
                        "variant": variant,
                        "feature": feature,
                        "feature_type": (
                            "numeric" if is_numeric
                            else "categorical"
                        ),
                        "training_status": status,
                        "train_records": len(train),
                        "train_valid_records": len(train_valid),
                        "train_missing_fraction": float(
                            train_values.isna().mean()
                        ),
                        "train_unique_values": int(unique_train),
                        "test_records": len(test),
                        "test_valid_records": len(test_valid),
                        "test_missing_fraction": float(
                            test_values.isna().mean()
                        ),
                        "test_unique_values": int(unique_test),
                        "unsupported_test_records": (
                            unsupported_count
                        ),
                        "unseen_category_records": unseen_count,
                        "constant_value_deviation_records": (
                            constant_deviation_count
                        ),
                    })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No feature eligibility records produced.")
    result_path = OUT / "feature_eligibility_by_fold.csv"
    result.to_csv(result_path, index=False)
    status_summary = (
        result.groupby(
            ["variant", "feature", "training_status"]
        )
        .size()
        .reset_index(name="fold_count")
    )
    status_path = OUT / "feature_eligibility_status_counts.csv"
    status_summary.to_csv(status_path, index=False)
    unsupported_summary = (
        result.groupby(
            ["target", "held_out_source", "variant"]
        )
        .agg(
            features=("feature", "count"),
            unsupported_feature_fold_pairs=(
                "unsupported_test_records",
                lambda x: int((x > 0).sum()),
            ),
            total_unsupported_test_records=(
                "unsupported_test_records", "sum"
            ),
            all_missing_training_features=(
                "training_status",
                lambda x: int(
                    (x == "ALL_MISSING_IN_TRAIN").sum()
                ),
            ),
            constant_training_features=(
                "training_status",
                lambda x: int(
                    (x == "CONSTANT_IN_TRAIN").sum()
                ),
            ),
        )
        .reset_index()
    )
    unsupported_path = (
        OUT / "fold_unsupported_feature_summary.csv"
    )
    unsupported_summary.to_csv(
        unsupported_path, index=False
    )
    report = {
        "experiment": (
            "MAT-X Applicability Feature Eligibility Diagnostic v3"
        ),
        "read_only": True,
        "source_dataset_modified": False,
        "models_modified": False,
        "fold_feature_rows": len(result),
        "targets": TARGETS,
        "variants": list(VARIANTS),
        "training_status_counts": {
            str(k): int(v)
            for k, v in result[
                "training_status"
            ].value_counts().items()
        },
        "limitations": [
            "Feature eligibility is based only on the fold training data.",
            "A constant training feature provides no learned variation.",
            "A feature entirely missing in training has no empirical reference.",
            "This diagnostic does not classify material records.",
            "No models were retrained or promoted.",
        ],
        "outputs": {
            "feature_eligibility": str(result_path),
            "status_counts": str(status_path),
            "fold_summary": str(unsupported_path),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("\nFEATURE ELIGIBILITY STATUS COUNTS")
    print(status_summary.to_string(index=False))
    print("\nFOLD-LEVEL UNSUPPORTED FEATURE SUMMARY")
    print(unsupported_summary.to_string(index=False))
    print("\nCOMPLETE")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
