import csv
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.integrations import steelbench_service as service
FIELDS = [
    "heat_id", "grade_id", "source",
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "quench_medium", "temper_T",
    "tensile_strength", "yield_strength", "elongation",
    "reduction_area", "impact_J_avg", "condition",
    "hardness", "split", "data_tier", "steel_family",
]
def write_fixture(path: Path, rows=None, fields=None):
    rows = rows or [
        {
            "heat_id": "H001", "grade_id": "4140", "source": "NIMS",
            "C": "0.40", "Mn": "0.80", "Cr": "1.0", "Mo": "0.20",
            "Ni": "", "Si": "0.25", "V": "", "Cu": "", "Al": "",
            "austenitize_T": "850", "quench_medium": "oil", "temper_T": "550",
            "tensile_strength": "950", "yield_strength": "750",
            "elongation": "16", "reduction_area": "45", "impact_J_avg": "35",
            "condition": "#NAME?", "hardness": "28", "split": "pretrain",
            "data_tier": "nims_measured", "steel_family": "CrMo_QT",
        },
        {
            "heat_id": "H002", "grade_id": "304", "source": "Kaggle_Stainless_Steel",
            "C": "0.04", "Mn": "1.2", "Cr": "18", "Mo": "0",
            "Ni": "8", "Si": "0.5", "V": "0", "Cu": "0", "Al": "0",
            "austenitize_T": "", "quench_medium": "", "temper_T": "",
            "tensile_strength": "620", "yield_strength": "290",
            "elongation": "40", "reduction_area": "55", "impact_J_avg": "",
            "condition": "????? ??????", "hardness": "",
            "split": "pretrain", "data_tier": "kaggle_measured",
            "steel_family": "stainless_austenitic",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or FIELDS)
        writer.writeheader()
        writer.writerows(rows)
def test_steelbench_loads_and_preserves_source_provenance(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    result = service.search_steelbench_records()
    assert result["status"] == "SUCCESS"
    assert result["dataset_version"] == "1.0"
    assert result["total_matches"] == 2
    assert result["records"][0]["heat_id"] == "H001"
    assert result["records"][0]["composition_wt_pct"]["C"] == 0.4
    assert result["records"][0]["mechanical_properties"]["tensile_strength_source_value"] == 950
    assert result["records"][0]["source_provenance"]["record_id"] == "H001"
    assert result["records"][0]["source_provenance"]["source_name"] == "NIMS"
    assert result["records"][0]["source_provenance"]["mechanical_property_unit_status"] == "UNVERIFIED"
def test_steelbench_preserves_messy_condition_values(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    result = service.search_steelbench_records(heat_id="H001")
    record = result["records"][0]
    assert record["condition_raw"] == "#NAME?"
    assert any("spreadsheet/error-like" in warning for warning in record["quality_warnings"])
    second = service.search_steelbench_records(heat_id="H002")
    assert second["records"][0]["condition_raw"] == "????? ??????"
def test_steelbench_missing_values_remain_none(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    record = service.search_steelbench_records(heat_id="H002")["records"][0]
    assert record["composition_wt_pct"]["Ni"] == 8
    assert record["mechanical_properties"]["impact_energy_J"] is None
    assert record["mechanical_properties"]["hardness_source_value"] is None
    assert record["process_history"]["austenitize_temperature_C"] is None
def test_steelbench_filters_and_paginates(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    filtered = service.search_steelbench_records(
        source="nims", steel_family="CrMo_QT"
    )
    assert filtered["total_matches"] == 1
    assert filtered["records"][0]["heat_id"] == "H001"
    page = service.search_steelbench_records(limit=1, offset=1)
    assert page["total_matches"] == 2
    assert page["record_count"] == 1
    assert page["records"][0]["heat_id"] == "H002"
def test_steelbench_text_search(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    result = service.search_steelbench_records(query="stainless")
    assert result["total_matches"] == 1
    assert result["records"][0]["grade_id"] == "304"
def test_steelbench_rejects_missing_columns(tmp_path, monkeypatch):
    path = tmp_path / "bad.csv"
    path.write_text("heat_id,grade_id,source\nH1,G1,S1\n", encoding="utf-8")
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    with pytest.raises(service.SteelBenchIntegrationError, match="missing required columns"):
        service.search_steelbench_records()
def test_steelbench_rejects_extra_csv_fields(tmp_path, monkeypatch):
    path = tmp_path / "bad.csv"
    write_fixture(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(["extra"] * (len(FIELDS) + 1)) + "\n")
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    with pytest.raises(service.SteelBenchIntegrationError, match="extra fields"):
        service.search_steelbench_records()
def test_steelbench_validates_pagination(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    with pytest.raises(ValueError, match="limit"):
        service.search_steelbench_records(limit=101)
    with pytest.raises(ValueError, match="offset"):
        service.search_steelbench_records(offset=-1)
def test_steelbench_api_route_and_openapi(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    from app.main import app
    client = TestClient(app)
    response = client.get("/api/v1/datasets/steelbench/records?limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "SteelBench"
    assert payload["record_count"] == 1
    assert payload["records"][0]["source_provenance"]["classification"] == "SOURCE_DATA_AS_PUBLISHED"
    openapi = client.get("/openapi.json").json()
    assert "/api/v1/datasets/steelbench/records" in openapi["paths"]
def test_steelbench_api_filter_and_validation(tmp_path, monkeypatch):
    path = tmp_path / "SteelBench.csv"
    write_fixture(path)
    monkeypatch.setattr(service, "STEELBENCH_PATH", path)
    from app.main import app
    client = TestClient(app)
    response = client.get(
        "/api/v1/datasets/steelbench/records",
        params={"source": "NIMS", "steel_family": "CrMo_QT"},
    )
    assert response.status_code == 200
    assert response.json()["total_matches"] == 1
    invalid = client.get("/api/v1/datasets/steelbench/records?limit=101")
    assert invalid.status_code == 422
