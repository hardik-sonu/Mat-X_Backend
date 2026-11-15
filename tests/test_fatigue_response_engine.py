import pandas as pd
import pytest
from app.engines import fatigue_response_engine as engine
def valid_cycles():
    frame = pd.DataFrame({
        "specimen_id": ["specimen_a", "specimen_a"],
        "geometry": ["flat", "flat"],
        "cycle_count": [1, 2],
        "declared_cycles": [2, 2],
        "stress_amplitude_MPa": [450.0, 440.0],
        "mean_stress_MPa": [-5.0, -4.0],
        "strain_amplitude": [0.004, 0.0041],
        "loop_area_MJ_m3": [1.5, 1.6],
        "source_file": ["source.csv", "source.csv"],
    })
    frame["cycle_count"] = frame["cycle_count"].astype(float)
    frame["declared_cycles"] = frame["declared_cycles"].astype(float)
    return frame
def run_with_frame(monkeypatch, tmp_path, frame):
    path = tmp_path / "cycles.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(engine, "CYCLE_FILE", path)
    monkeypatch.setattr(engine, "DATA_DIR", tmp_path)
    return engine.analyze_fatigue_response(
        write_outputs=False,
        verbose=False,
    )
def test_valid_fatigue_records_are_analyzed(monkeypatch, tmp_path):
    report = run_with_frame(monkeypatch, tmp_path, valid_cycles())
    assert report["cycle_feature_records"] == 2
    assert report["independent_specimens"] == 1
    assert report["status"] == (
        "RESPONSE_ANALYSIS_ONLY_NOT_FATIGUE_LIFE_VALIDATED"
    )
@pytest.mark.parametrize("bad_value", [0, -1, 1.5, float("inf")])
def test_invalid_cycle_counts_are_rejected(
    monkeypatch, tmp_path, bad_value
):
    frame = valid_cycles()
    frame.loc[0, "cycle_count"] = bad_value
    with pytest.raises(ValueError):
        run_with_frame(monkeypatch, tmp_path, frame)
@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("stress_amplitude_MPa", float("inf")),
        ("stress_amplitude_MPa", 0),
        ("strain_amplitude", float("nan")),
        ("strain_amplitude", -0.1),
        ("mean_stress_MPa", float("-inf")),
        ("declared_cycles", 0),
        ("declared_cycles", 2.5),
    ],
)
def test_invalid_physical_or_declared_values_are_rejected(
    monkeypatch, tmp_path, column, bad_value
):
    frame = valid_cycles()
    frame.loc[0, column] = bad_value
    with pytest.raises(ValueError):
        run_with_frame(monkeypatch, tmp_path, frame)
def test_inconsistent_specimen_geometry_is_rejected(monkeypatch, tmp_path):
    frame = valid_cycles()
    frame.loc[1, "geometry"] = "cylindrical"
    with pytest.raises(ValueError, match="Inconsistent geometry"):
        run_with_frame(monkeypatch, tmp_path, frame)
def test_inconsistent_declared_cycles_are_rejected(monkeypatch, tmp_path):
    frame = valid_cycles()
    frame.loc[1, "declared_cycles"] = 3
    with pytest.raises(ValueError, match="Inconsistent declared"):
        run_with_frame(monkeypatch, tmp_path, frame)
def test_recorded_cycles_cannot_exceed_declared_total(monkeypatch, tmp_path):
    frame = valid_cycles()
    frame["declared_cycles"] = 1.0
    with pytest.raises(ValueError, match="exceeds declared total"):
        run_with_frame(monkeypatch, tmp_path, frame)
