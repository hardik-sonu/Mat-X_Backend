from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "training_results" / "fatigue"
CYCLE_FILE = DATA_DIR / "fatigue_316L_cycle_features.csv"
SUMMARY_FILE = DATA_DIR / "fatigue_316L_specimen_summary.csv"
def _safe_float(value):
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None
def analyze_fatigue_response(write_outputs=True, verbose=True):
    if not CYCLE_FILE.exists():
        raise FileNotFoundError(f"Missing fatigue cycle file: {CYCLE_FILE}")
    cycles = pd.read_csv(CYCLE_FILE)
    required = {
        "specimen_id",
        "geometry",
        "cycle_count",
        "stress_amplitude_MPa",
        "mean_stress_MPa",
        "strain_amplitude",
        "source_file",
    }
    missing = sorted(required - set(cycles.columns))
    if missing:
        raise ValueError(f"Missing required cycle-feature columns: {missing}")
    if cycles.empty:
        raise ValueError("Fatigue cycle-feature dataset is empty.")
    numeric_columns = [
        "cycle_count",
        "stress_amplitude_MPa",
        "mean_stress_MPa",
        "strain_amplitude",
    ]
    if "declared_cycles" in cycles.columns:
        numeric_columns.append("declared_cycles")
    if "loop_area_MJ_m3" in cycles.columns:
        numeric_columns.append("loop_area_MJ_m3")
    for column in numeric_columns:
        cycles[column] = pd.to_numeric(cycles[column], errors="coerce")
        values = cycles[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Fatigue column '{column}' contains missing or non-finite values."
            )
    for column in ["specimen_id", "geometry", "source_file"]:
        values = cycles[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Fatigue column '{column}' contains missing or blank values."
            )
    cycle_values = cycles["cycle_count"].to_numpy(dtype=float)
    if (cycle_values < 1).any() or not np.equal(
        cycle_values, np.floor(cycle_values)
    ).all():
        raise ValueError("Cycle counts must be positive whole numbers.")
    if (cycles["stress_amplitude_MPa"] <= 0).any():
        raise ValueError("Stress amplitude must be greater than zero.")
    if (cycles["strain_amplitude"] <= 0).any():
        raise ValueError("Strain amplitude must be greater than zero.")
    if "declared_cycles" in cycles.columns:
        declared_values = cycles["declared_cycles"].to_numpy(dtype=float)
        if (declared_values < 1).any() or not np.equal(
            declared_values, np.floor(declared_values)
        ).all():
            raise ValueError(
                "Declared cycle totals must be positive whole numbers."
            )
    results = []
    results = []
    for specimen_id, group in cycles.groupby("specimen_id", sort=True):
        group = group.sort_values("cycle_count").copy()
        if group["geometry"].nunique(dropna=False) != 1:
            raise ValueError(
                f"Inconsistent geometry values for specimen {specimen_id}."
            )
        if "declared_cycles" in group.columns:
            if group["declared_cycles"].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Inconsistent declared cycle totals for specimen {specimen_id}."
                )
            declared_total = float(group["declared_cycles"].iloc[0])
            if (group["cycle_count"] > declared_total).any():
                raise ValueError(
                    f"Recorded cycle count exceeds declared total for specimen {specimen_id}."
                )
        if group["cycle_count"].duplicated().any():
            raise ValueError(
                f"Duplicate cycle counts found for specimen {specimen_id}."
            )
        record = {
            "specimen_id": str(specimen_id),
            "geometry": str(group["geometry"].iloc[0]),
            "recorded_cycle_count": int(len(group)),
            "first_recorded_cycle": int(group["cycle_count"].min()),
            "last_recorded_cycle": int(group["cycle_count"].max()),
            "sampling_status": "SPARSELY_SAMPLED",
            "independent_specimen": True,
        }
        declared = None
        if "declared_cycles" in group.columns:
            values = pd.to_numeric(
                group["declared_cycles"], errors="coerce"
            ).dropna()
            if not values.empty:
                declared = int(values.max())
        record["declared_cycles"] = declared
        record["cycle_coverage_fraction"] = (
            _safe_float(
                group["cycle_count"].nunique() / declared
            )
            if declared and declared > 0 else None
        )
        for column in [
            "stress_amplitude_MPa",
            "mean_stress_MPa",
            "strain_amplitude",
            "loop_area_MJ_m3",
        ]:
            if column not in group.columns:
                continue
            values = group[column].dropna()
            record[f"{column}_mean"] = (
                _safe_float(values.mean()) if not values.empty else None
            )
            record[f"{column}_initial"] = (
                _safe_float(values.iloc[0]) if not values.empty else None
            )
            record[f"{column}_final"] = (
                _safe_float(values.iloc[-1]) if not values.empty else None
            )
            if len(values) >= 2:
                record[f"{column}_min"] = _safe_float(values.min())
                record[f"{column}_max"] = _safe_float(values.max())
        stress = group[["cycle_count", "stress_amplitude_MPa"]].dropna()
        if len(stress) >= 2:
            slope = np.polyfit(
                stress["cycle_count"].to_numpy(dtype=float),
                stress["stress_amplitude_MPa"].to_numpy(dtype=float),
                1,
            )[0]
            record["stress_amplitude_trend_MPa_per_cycle"] = _safe_float(slope)
        else:
            record["stress_amplitude_trend_MPa_per_cycle"] = None
        record["source_files"] = sorted(
            group["source_file"].dropna().astype(str).unique().tolist()
        )
        record["loop_area_status"] = (
            "PROVISIONAL_DERIVED_VALUE"
            if "loop_area_MJ_m3" in group.columns
            else "UNAVAILABLE"
        )
        results.append(record)
    specimen_results = pd.DataFrame(results)
    report = {
        "engine": "MAT-X 316L Fatigue Response Analysis",
        "engine_version": "1.0.0",
        "source_file": CYCLE_FILE.name,
        "cycle_feature_records": int(len(cycles)),
        "independent_specimens": int(cycles["specimen_id"].nunique()),
        "geometry_groups": sorted(
            cycles["geometry"].dropna().astype(str).unique().tolist()
        ),
        "specimen_results": specimen_results.to_dict(orient="records"),
        "scientific_scope": [
            "Descriptive analysis of recorded fatigue response.",
            "Results are grouped by independent specimen.",
            "Sparse cycle observations are not interpolated.",
            "Loop area remains a provisional derived quantity.",
            "No generalizable fatigue-life prediction is claimed.",
        ],
        "status": "RESPONSE_ANALYSIS_ONLY_NOT_FATIGUE_LIFE_VALIDATED",
    }
    if verbose:
        print("FATIGUE RESPONSE ENGINE COMPLETED")
        print("Cycle records:", len(cycles))
        print("Independent specimens:", cycles["specimen_id"].nunique())
        for item in results:
            print(
                item["specimen_id"],
                "| cycles:", item["recorded_cycle_count"],
                "| coverage:", item["cycle_coverage_fraction"],
                "| stress amplitude trend:",
                item["stress_amplitude_trend_MPa_per_cycle"],
            )
    if write_outputs:
        output_csv = DATA_DIR / "fatigue_response_analysis.csv"
        output_json = DATA_DIR / "fatigue_response_analysis.json"
        specimen_results.to_csv(output_csv, index=False)
        import json
        output_json.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return report
if __name__ == "__main__":
    analyze_fatigue_response()
