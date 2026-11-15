import numpy as np
import pandas as pd
SOURCE_REQUIRED_COLUMNS = [
    "heat_id",
    "grade_id",
    "steel_family",
    "source",
    "austenitize_T",
    "quench_medium",
    "temper_T",
]
GENERATED_REQUIRED_COLUMNS = SOURCE_REQUIRED_COLUMNS + [
    "documented_field_count",
    "missing_process_fields",
    "process_status",
    "documented_route_evidence",
    "provenance",
]
PROCESS_FIELDS = ["austenitize_T", "quench_medium", "temper_T"]
STATUS_BY_COUNT = {
    0: "NO_DOCUMENTED_PROCESS_HISTORY",
    1: "PARTIAL_PROCESS_HISTORY",
    2: "PARTIAL_PROCESS_HISTORY",
    3: "COMPLETE_DOCUMENTED_PROCESS_FIELDS",
}
def validate_process_history_frame(df, require_generated=False):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Process history data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Process history dataset is empty.")
    required = (
        GENERATED_REQUIRED_COLUMNS
        if require_generated
        else SOURCE_REQUIRED_COLUMNS
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required process history columns: {missing}")
    for column in ["heat_id", "grade_id", "steel_family", "source"]:
        values = df[column]
        invalid = values.isna() | values.astype("string").str.strip().eq("").fillna(False)
        if invalid.any():
            raise ValueError(
                f"Missing or blank process history identifiers in {column}."
            )
    if df["heat_id"].duplicated().any():
        raise ValueError("Duplicate heat_id values found in process history data.")
    for column in ["austenitize_T", "temper_T"]:
        original = df[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Non-numeric values found in {column}.")
        finite = numeric.dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise ValueError(f"Non-finite values found in {column}.")
    quench = df["quench_medium"]
    blank_quench = (
        quench.notna()
        & quench.astype("string").str.strip().eq("").fillna(False)
    )
    if blank_quench.any():
        raise ValueError("Blank documented quench_medium values are invalid.")
    if "process_status" in df.columns:
        allowed = set(STATUS_BY_COUNT.values())
        unexpected = set(df["process_status"].dropna().unique()) - allowed
        if unexpected:
            raise ValueError(
                f"Unknown process_status values: {sorted(unexpected)}"
            )
        if df["process_status"].isna().any():
            raise ValueError("Missing process_status values found.")
        counts = df[PROCESS_FIELDS].notna().sum(axis=1)
        expected_status = counts.map(STATUS_BY_COUNT)
        mismatch = df["process_status"].astype(str) != expected_status
        if mismatch.any():
            raise ValueError(
                "process_status does not match the documented process fields."
            )
        if "documented_field_count" in df.columns:
            documented = pd.to_numeric(
                df["documented_field_count"], errors="coerce"
            )
            if documented.isna().any() or not documented.eq(counts).all():
                raise ValueError(
                    "documented_field_count does not match observed process fields."
                )
    return df
