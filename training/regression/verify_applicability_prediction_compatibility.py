from pathlib import Path
import pandas as pd
ROOT = Path("training_results")
ABL = ROOT / "controlled_feature_ablation_v1"
AD = ROOT / "applicability_domain_v3_retry"
print("\n=== 1. SPLIT AUDIT COMPARISON ===")
a = pd.read_csv(ABL / "shared_split_audit.csv")
b = pd.read_csv(AD / "split_integrity_audit.csv")
keys = ["target", "held_out_source"]
cols = [
    "train_records", "calibration_records", "test_records",
    "train_grades", "calibration_grades", "test_grades",
    "train_calibration_overlap", "train_test_overlap",
    "calibration_test_overlap",
]
m = a.merge(b, on=keys, how="outer",
            suffixes=("_abl", "_app"), indicator=True)
diff_rows = m["_merge"].ne("both")
for c in cols:
    left, right = c + "_abl", c + "_app"
    equal = m[left].eq(m[right]) | (m[left].isna() & m[right].isna())
    diff_rows |= ~equal
    print(f"{c}: {int(equal.sum())}/{len(m)} equal")
print("\nAudit row differences:")
print(m.loc[diff_rows].to_string(index=False) if diff_rows.any()
      else "NONE")
print("\n=== 2. PREDICTION INVENTORY ===")
pred_root = ABL / "predictions"
summary = []
for path in sorted(pred_root.rglob("*.csv")):
    parts = path.relative_to(pred_root).parts
    if len(parts) != 3:
        print("UNEXPECTED PATH:", path)
        continue
    target, held_out, variant = parts
    df = pd.read_csv(path)
    required = {
        "source", "grade_id", "heat_id",
        "actual", "prediction", "absolute_error",
    }
    missing = required - set(df.columns)
    key_cols = ["source", "grade_id", "heat_id"]
    duplicates = (
        int(df.duplicated(key_cols).sum())
        if not missing else -1
    )
    summary.append({
        "target": target,
        "held_out_source": held_out,
        "variant": variant,
        "rows": len(df),
        "unique_heats": df["heat_id"].nunique()
            if "heat_id" in df else -1,
        "duplicate_keys": duplicates,
        "missing_columns": ",".join(sorted(missing)),
    })
s = pd.DataFrame(summary)
if not s.empty:
    print(s.to_string(index=False))
    print("\nPrediction files:", len(s))
    print("Prediction rows:", int(s["rows"].sum()))
    print("Duplicate keys:", int(s["duplicate_keys"].sum()))
print("\n=== 3. APPLICABILITY INVENTORY ===")
app = pd.read_csv(AD / "record_applicability.csv")
print("Applicability rows:", len(app))
print(
    app.groupby(["target", "held_out_source", "variant"])
       .size().to_string()
)
print("\n=== 4. SCRIPT LOGIC ===")
scripts = [
    Path("training/regression/run_controlled_feature_ablation.py"),
    Path("training/regression/run_applicability_domain_assessment_v3.py"),
]
terms = [
    "GroupShuffleSplit", "random_state", "test_size",
    "calibration", "held_out", "dropna", "ExtraTrees",
    "n_estimators", "min_samples_leaf", "FEATURES",
    "FEATURE_COLUMNS", "TARGET_COLUMNS",
]
for path in scripts:
    print(f"\n--- {path} ---")
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if any(t.lower() in line.lower() for t in terms):
            print(f"{i:04}: {line[:240]}")
print("\n=== COMPATIBILITY INSPECTION COMPLETE ===")
