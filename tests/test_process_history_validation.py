import pandas as pd
import pytest
from app.engines.process_history_validation import (
    validate_process_history_frame,
)
def valid_frame():
    return pd.DataFrame([
        {
            "heat_id": "H-001",
            "grade_id": "G-001",
            "steel_family": "low_alloy_steel",
            "source": "dataset_a",
            "austenitize_T": 900.0,
            "quench_medium": "water",
            "temper_T": None,
            "documented_field_count": 2,
            "missing_process_fields": "['temper_T']",
            "process_status": "PARTIAL_PROCESS_HISTORY",
            "documented_route_evidence": "[]",
            "provenance": "{}",
        }
    ])
def test_valid_process_history_preserves_missing_process_fields():
    df = valid_frame()
    result = validate_process_history_frame(df, require_generated=True)
    assert len(result) == 1
    assert pd.isna(result.loc[0, "temper_T"])
    assert result.loc[0, "process_status"] == "PARTIAL_PROCESS_HISTORY"
@pytest.mark.parametrize("column", ["heat_id", "grade_id", "steel_family", "source"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_rejects_missing_or_blank_identifiers(column, value):
    df = valid_frame()
    df.loc[0, column] = value
    with pytest.raises(ValueError, match="identifiers"):
        validate_process_history_frame(df, require_generated=True)
def test_rejects_duplicate_heat_ids():
    df = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate heat_id"):
        validate_process_history_frame(df, require_generated=True)
@pytest.mark.parametrize("value", ["not-a-temperature", float("inf"), float("-inf")])
@pytest.mark.parametrize("column", ["austenitize_T", "temper_T"])
def test_rejects_invalid_documented_temperatures(column, value):
    df = valid_frame()
    df[column] = df[column].astype(object)
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_process_history_frame(df, require_generated=True)
def test_rejects_blank_documented_quench_medium():
    df = valid_frame()
    df.loc[0, "quench_medium"] = "   "
    with pytest.raises(ValueError, match="quench_medium"):
        validate_process_history_frame(df, require_generated=True)
def test_rejects_inconsistent_process_status():
    df = valid_frame()
    df.loc[0, "process_status"] = "COMPLETE_DOCUMENTED_PROCESS_FIELDS"
    with pytest.raises(ValueError, match="process_status"):
        validate_process_history_frame(df, require_generated=True)
def test_rejects_missing_generated_columns():
    df = valid_frame().drop(columns=["provenance"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_process_history_frame(df, require_generated=True)
