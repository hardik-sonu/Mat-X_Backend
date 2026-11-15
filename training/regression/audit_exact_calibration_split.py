import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
RANDOM_STATE = 42
TEST_FRACTION = 0.20
CALIBRATION_FRACTION = 0.25
NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T"
]
CATEGORICAL = ["quench_medium", "steel_family"]
FEATURES = NUMERIC + CATEGORICAL
def promoted_split(df, target):
    work = df.dropna(subset=[target, "grade_id"]).copy()
    if target == "yield_strength":
        work["YS_UTS_ratio"] = (
            work[target]
            / work["tensile_strength"].replace(0, float("nan"))
        )
        anomaly = (
            work["grade_id"].astype(str).isin(
                ["EMK0319", "EMK0320", "EMK0321"]
            )
            & (work["YS_UTS_ratio"] < 0.03)
        )
        work = work.loc[~anomaly].copy()
    X = work[FEATURES]
    y = work[target].astype(float)
    groups = work["grade_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )
    return (
        set(work.index[train_idx]),
        set(work.index[test_idx]),
    )
def calibration_split(df, target):
    work = df.copy()
    work[target] = pd.to_numeric(
        work[target], errors="coerce"
    )
    work = work.dropna(subset=[target, "grade_id"]).copy()
    work["grade_id"] = work["grade_id"].astype(str)
    for col in NUMERIC:
        work[col] = pd.to_numeric(
            work[col], errors="coerce"
        )
    if target == "yield_strength":
        work["tensile_strength"] = pd.to_numeric(
            work["tensile_strength"], errors="coerce"
        )
        ratio = (
            work[target]
            / work["tensile_strength"].replace(0, float("nan"))
        )
        anomaly = (
            work["grade_id"].isin(
                ["EMK0319", "EMK0320", "EMK0321"]
            )
            & (ratio < 0.03)
        )
        work = work.loc[~anomaly].copy()
    X = work[FEATURES]
    y = work[target].astype(float)
    groups = work["grade_id"]
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_cal_idx, test_idx = next(
        first.split(X, y, groups=groups)
    )
    X_train_cal = X.iloc[train_cal_idx]
    y_train_cal = y.iloc[train_cal_idx]
    groups_train_cal = groups.iloc[train_cal_idx]
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE + 1,
    )
    train_rel, calibration_rel = next(
        second.split(
            X_train_cal,
            y_train_cal,
            groups=groups_train_cal,
        )
    )
    return (
        set(work.index[train_idx])
        if False else set(work.index[train_cal_idx[train_rel]]),
        set(work.index[train_cal_idx[calibration_rel]]),
        set(work.index[test_idx]),
    )
df = pd.read_csv(DATASET, low_memory=False)
print("Dataset rows:", len(df))
print("Dataset columns:", len(df.columns))
for target in ["tensile_strength", "yield_strength"]:
    print("\n" + "=" * 72)
    print(target.upper())
    print("=" * 72)
    promoted_train, promoted_test = promoted_split(df, target)
    cal_train, cal_calibration, cal_test = calibration_split(
        df, target
    )
    print("Promoted training rows:", len(promoted_train))
    print("Calibration training rows:", len(cal_train))
    print("Calibration calibration rows:", len(cal_calibration))
    print("Promoted test rows:", len(promoted_test))
    print("Calibration test rows:", len(cal_test))
    print("Exact test row-index match:", promoted_test == cal_test)
    print("Promoted-only test rows:", len(promoted_test - cal_test))
    print("Calibration-only test rows:", len(cal_test - promoted_test))
    print(
        "Test grade sets match:",
        set(df.loc[list(promoted_test), "grade_id"].astype(str))
        == set(df.loc[list(cal_test), "grade_id"].astype(str))
    )
    print("Training/test overlap in calibration:",
          bool(cal_train & cal_test))
    print("Calibration/test overlap:",
          bool(cal_calibration & cal_test))
    print("Training/calibration overlap:",
          bool(cal_train & cal_calibration))
print("\nROW-LEVEL SPLIT AUDIT COMPLETED")
