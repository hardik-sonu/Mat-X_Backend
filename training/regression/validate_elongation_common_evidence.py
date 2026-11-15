from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "regression"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [
    "EMK_spec",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
BASE_FEATURES = ["C", "Mn", "V", "steel_family"]
ADDED_EVIDENCE = ["tensile_strength", "yield_strength"]
ENRICHED_FEATURES = BASE_FEATURES + ADDED_EVIDENCE
df = pd.read_csv(DATA)
for col in ["elongation", "C", "Mn", "V"] + ADDED_EVIDENCE:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[
    df["source"].isin(SOURCES) & df["elongation"].notna()
].copy().reset_index(drop=True)
for feature in BASE_FEATURES + ADDED_EVIDENCE:
    missing = df.groupby("source")[feature].apply(
        lambda s: int(s.isna().sum())
    )
    if missing.any():
        raise ValueError(
            f"{feature} is not complete across sources:\n{missing}"
        )
def build_model(features):
    numeric = [f for f in features if f != "steel_family"]
    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            numeric,
        ),
        (
            "family",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            ["steel_family"],
        ),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
logo = LeaveOneGroupOut()
predictions = []
fold_reports = []
for fold, (train_idx, test_idx) in enumerate(
    logo.split(df, df["elongation"], groups=df["source"]),
    start=1,
):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    held_out = test["source"].iloc[0]
    train_sources = sorted(train["source"].unique())
    grade_overlap = set(train["grade_id"]) & set(test["grade_id"])
    assert not grade_overlap
    assert held_out not in train_sources
    baseline = build_model(BASE_FEATURES)
    enriched = build_model(ENRICHED_FEATURES)
    baseline.fit(train[BASE_FEATURES], train["elongation"])
    enriched.fit(train[ENRICHED_FEATURES], train["elongation"])
    actual = test["elongation"].to_numpy()
    bp = baseline.predict(test[BASE_FEATURES])
    ep = enriched.predict(test[ENRICHED_FEATURES])
    b_mae = mean_absolute_error(actual, bp)
    e_mae = mean_absolute_error(actual, ep)
    b_rmse = np.sqrt(mean_squared_error(actual, bp))
    e_rmse = np.sqrt(mean_squared_error(actual, ep))
    fold_reports.append({
        "fold": fold,
        "held_out_source": held_out,
        "training_sources": train_sources,
        "train_records": len(train),
        "test_records": len(test),
        "grade_overlap": len(grade_overlap),
        "baseline": {
            "mae": float(b_mae),
            "rmse": float(b_rmse),
            "r2": float(r2_score(actual, bp)),
        },
        "common_evidence_enriched": {
            "mae": float(e_mae),
            "rmse": float(e_rmse),
            "r2": float(r2_score(actual, ep)),
        },
        "mae_improvement_percent": float(
            (b_mae - e_mae) / b_mae * 100
        ),
        "rmse_improvement_percent": float(
            (b_rmse - e_rmse) / b_rmse * 100
        ),
    })
    for i, (_, row) in enumerate(test.iterrows()):
        predictions.append({
            "source": held_out,
            "grade_id": row["grade_id"],
            "heat_id": row["heat_id"],
            "actual": float(actual[i]),
            "baseline_prediction": float(bp[i]),
            "enriched_prediction": float(ep[i]),
            "baseline_absolute_error": float(abs(actual[i] - bp[i])),
            "enriched_absolute_error": float(abs(actual[i] - ep[i])),
        })
    print("\n" + "=" * 72)
    print(f"HELD-OUT SOURCE: {held_out}")
    print(f"Training sources: {train_sources}")
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Grade overlap: {len(grade_overlap)}")
    print("\n                 BASELINE     ENRICHED")
    print(f"MAE              {b_mae:9.4f}    {e_mae:9.4f}")
    print(f"RMSE             {b_rmse:9.4f}    {e_rmse:9.4f}")
    print(
        f"R2               {r2_score(actual, bp):9.4f}"
        f"    {r2_score(actual, ep):9.4f}"
    )
    print(f"MAE improvement: {(b_mae-e_mae)/b_mae*100:.2f}%")
pred = pd.DataFrame(predictions)
assert len(pred) == len(df)
assert pred["source"].nunique() == 3
actual = pred["actual"].to_numpy()
bp = pred["baseline_prediction"].to_numpy()
ep = pred["enriched_prediction"].to_numpy()
pooled = {
    "records": len(pred),
    "baseline": {
        "mae": float(mean_absolute_error(actual, bp)),
        "rmse": float(np.sqrt(mean_squared_error(actual, bp))),
        "r2": float(r2_score(actual, bp)),
    },
    "common_evidence_enriched": {
        "mae": float(mean_absolute_error(actual, ep)),
        "rmse": float(np.sqrt(mean_squared_error(actual, ep))),
        "r2": float(r2_score(actual, ep)),
    },
}
pooled["mae_improvement_percent"] = (
    (pooled["baseline"]["mae"] -
     pooled["common_evidence_enriched"]["mae"])
    / pooled["baseline"]["mae"] * 100
)
pooled["rmse_improvement_percent"] = (
    (pooled["baseline"]["rmse"] -
     pooled["common_evidence_enriched"]["rmse"])
    / pooled["baseline"]["rmse"] * 100
)
report = {
    "experiment": "Elongation common-evidence source-held-out validation",
    "validation": "LeaveOneGroupOut by source",
    "baseline_features": BASE_FEATURES,
    "added_evidence": ADDED_EVIDENCE,
    "source_ids_used_as_features": False,
    "grade_ids_used_as_features": False,
    "original_dataset_modified": False,
    "model_promoted": False,
    "folds": fold_reports,
    "pooled": pooled,
}
pred_path = OUT / "matx_elongation_common_evidence_predictions.csv"
report_path = OUT / "matx_elongation_common_evidence_report.json"
pred.to_csv(pred_path, index=False)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n" + "=" * 72)
print("POOLED SOURCE-HELD-OUT RESULTS")
print("=" * 72)
print("Records:", pooled["records"])
print("Baseline:", pooled["baseline"])
print("Common evidence enriched:", pooled["common_evidence_enriched"])
print(f"MAE improvement: {pooled['mae_improvement_percent']:.2f}%")
print(f"RMSE improvement: {pooled['rmse_improvement_percent']:.2f}%")
print("\nSAVED FILES:")
print(pred_path)
print(report_path)
print("\nVALIDATION COMPLETE. NO MODEL PROMOTED.")
