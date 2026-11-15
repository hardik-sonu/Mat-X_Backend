import csv
import pytest
from app.engines.active_investigation_experiment_evidence import (
    evaluate_experiment_g,
)
def _write_csv(path, rows):
    columns = [
        "target",
        "composition_error_improvement",
        "process_error_improvement",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
def test_experiment_g_summarizes_paired_evidence_by_target(tmp_path):
    path = tmp_path / "experiment_g.csv"
    _write_csv(path, [
        {"target": "tensile", "composition_error_improvement": 10,
         "process_error_improvement": 2},
        {"target": "tensile", "composition_error_improvement": 5,
         "process_error_improvement": -1},
        {"target": "yield", "composition_error_improvement": -2,
         "process_error_improvement": 3},
    ])
    result = evaluate_experiment_g(path)
    assert result["status"] == "LIMITED_EXPERIMENTAL_PREDICTION_ERROR_ABLATION"
    assert result["scope"]["measures_information_gain"] is False
    assert result["scope"]["validates_physical_test_acquisition"] is False
    assert result["targets"]["tensile"]["composition"]["records"] == 2
    assert result["targets"]["tensile"]["composition"]["improved_records"] == 2
    assert result["targets"]["tensile"]["descriptive_preference"] == "COMPOSITION"
    assert result["targets"]["yield"]["descriptive_preference"] == "PROCESS"
def test_experiment_g_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("target,composition_error_improvement\ntensile,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_experiment_g(path)
def test_experiment_g_rejects_non_finite_values(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(path, [
        {"target": "tensile", "composition_error_improvement": "nan",
         "process_error_improvement": 1},
    ])
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate_experiment_g(path)
def test_experiment_g_rejects_empty_dataset(tmp_path):
    path = tmp_path / "empty.csv"
    _write_csv(path, [])
    with pytest.raises(ValueError, match="no data records"):
        evaluate_experiment_g(path)
