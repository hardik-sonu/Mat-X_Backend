import pandas as pd
from app.services import process_history_service as service

def fixture_fields(heat, grade, aust, quench, temper, family="steel", source="test"):
    values = (aust, quench, temper)
    count = sum(value is not None for value in values)
    status = (
        "COMPLETE_DOCUMENTED_PROCESS_FIELDS" if count == 3
        else "NO_DOCUMENTED_PROCESS_HISTORY" if count == 0
        else "PARTIAL_PROCESS_HISTORY"
    )
    return {
        "heat_id": heat,
        "grade_id": grade,
        "steel_family": family,
        "source": source,
        "austenitize_T": aust,
        "quench_medium": quench,
        "temper_T": temper,
        "documented_field_count": count,
        "missing_process_fields": "[]",
        "process_status": status,
        "documented_route_evidence": "[]",
        "provenance": "{}",
    }
def test_heat_lookup_reconstructs_missing_process_history_csv(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "process_history"
    monkeypatch.setattr(service, "OUTPUT", output_dir)
    calls = []
    def create_reconstruction():
        calls.append(True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [fixture_fields(
                "HEAT-COLD-START", "GRADE-COLD-START",
                900.0, "water", None,
            )]
        ).to_csv(
            output_dir / "process_history_reconstruction.csv",
            index=False,
        )
    monkeypatch.setattr(
        service, "reconstruct_process_history", create_reconstruction
    )
    result = service.get_heat_process_history("HEAT-COLD-START")
    assert len(calls) == 1
    assert result["record_count"] == 1
    assert result["records"][0]["grade_id"] == "GRADE-COLD-START"
    assert (
        output_dir / "process_history_reconstruction.csv"
    ).is_file()
def test_existing_process_history_csv_does_not_trigger_reconstruction(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "process_history"
    output_dir.mkdir()
    pd.DataFrame(
        [fixture_fields(
            "HEAT-EXISTING", "GRADE-EXISTING",
            850.0, "oil", 550.0,
        )]
    ).to_csv(
        output_dir / "process_history_reconstruction.csv",
        index=False,
    )
    monkeypatch.setattr(service, "OUTPUT", output_dir)
    def unexpected_reconstruction():
        raise AssertionError("Existing process history must be reused.")
    monkeypatch.setattr(
        service, "reconstruct_process_history", unexpected_reconstruction
    )
    result = service.get_grade_process_history("GRADE-EXISTING")
    assert result["record_count"] == 1
    assert result["records"][0]["heat_id"] == "HEAT-EXISTING"
