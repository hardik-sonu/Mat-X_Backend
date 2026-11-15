from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "regression"
REPORT = OUT / "evidence_enriched_validation_audit.txt"
df = pd.read_csv(DATA)
properties = [
    "tensile_strength",
    "yield_strength",
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
targets = [
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
lines = []
def log(value=""):
    print(value)
    lines.append(str(value))
def metrics(frame, prediction_column):
    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    errors = np.abs(actual - predicted)
    if len(frame) == 0:
        return None
    return {
        "n": len(frame),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }
log("=" * 100)
log("MAT-X EVIDENCE-ENRICHED VALIDATION AUDIT")
log("=" * 100)
if df["heat_id"].duplicated().any():
    raise ValueError("Dataset heat_id is not unique; safe merge stopped.")
for target in targets:
    path = OUT / (
        f"matx_{target}_evidence_enriched_test_predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path)
    evidence = [p for p in properties if p != target]
    metadata = df[
        ["heat_id", "grade_id", "source"] + evidence
    ].copy()
    merged = pred.merge(
        metadata,
        on=["heat_id", "grade_id", "source"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise AssertionError(
            f"{target}: some test records did not match original dataset."
        )
    merged.drop(columns="_merge", inplace=True)
    merged["available_evidence_count"] = (
        merged[evidence].notna().sum(axis=1)
    )
    log("\n" + "=" * 100)
    log(f"TARGET: {target}")
    log("=" * 100)
    log("\nOVERALL TEST PERFORMANCE")
    for name, col in [
        ("BASELINE", "baseline_prediction"),
        ("ENRICHED", "enriched_prediction"),
    ]:
        result = metrics(merged, col)
        log(
            f"{name:<12} N={result['n']:>3} "
            f"MAE={result['mae']:.5f} "
            f"RMSE={result['rmse']:.5f}"
        )
    log("\nSOURCE-WISE PERFORMANCE")
    for source, group in merged.groupby("source"):
        b = metrics(group, "baseline_prediction")
        e = metrics(group, "enriched_prediction")
        log(
            f"{source}: N={len(group)} | "
            f"Baseline MAE={b['mae']:.5f} | "
            f"Enriched MAE={e['mae']:.5f} | "
            f"MAE improvement={b['mae'] - e['mae']:.5f}"
        )
    log("\nTEST-SET ADDITIONAL EVIDENCE COVERAGE")
    for count, group in merged.groupby("available_evidence_count"):
        log(
            f"Available properties={count}/{len(evidence)} | "
            f"N={len(group)}"
        )
    log("\nCOMPLETE VS INCOMPLETE EVIDENCE")
    complete = merged[
        merged["available_evidence_count"] == len(evidence)
    ]
    incomplete = merged[
        merged["available_evidence_count"] < len(evidence)
    ]
    for label, group in [
        ("COMPLETE", complete),
        ("INCOMPLETE", incomplete),
    ]:
        if len(group) == 0:
            log(f"{label}: no test records")
            continue
        b = metrics(group, "baseline_prediction")
        e = metrics(group, "enriched_prediction")
        log(
            f"{label}: N={len(group)} | "
            f"Baseline MAE={b['mae']:.5f} | "
            f"Enriched MAE={e['mae']:.5f} | "
            f"Improvement={b['mae'] - e['mae']:.5f}"
        )
    merged["paired_error_improvement"] = (
        merged["baseline_absolute_error"]
        - merged["enriched_absolute_error"]
    )
    log("\nPAIRED TEST ERROR")
    log(
        "Records where enriched model has lower absolute error: "
        f"{int((merged['paired_error_improvement'] > 0).sum())}"
    )
    log(
        "Records where baseline model has lower absolute error: "
        f"{int((merged['paired_error_improvement'] < 0).sum())}"
    )
    log(
        "Records with equal absolute error: "
        f"{int((merged['paired_error_improvement'] == 0).sum())}"
    )
    log(
        "Mean paired absolute-error improvement: "
        f"{merged['paired_error_improvement'].mean():.5f}"
    )
log("\n" + "=" * 100)
log("VALIDATION AUDIT COMPLETED")
log("Original dataset and model artifacts were not modified.")
log("=" * 100)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print("\nReport saved:", REPORT)
