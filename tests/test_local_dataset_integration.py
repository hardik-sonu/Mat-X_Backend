import csv
from pathlib import Path
import pytest
from app.integrations import local_dataset_service as service
def write_csv(path: Path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grade", "carbon", "source"],
        )
        writer.writeheader()
        writer.writerow({
            "grade": "AISI 1045",
            "carbon": "0.45",
            "source": "laboratory",
        })
        writer.writerow({
            "grade": "AISI 4140",
            "carbon": "0.40",
            "source": "SteelBench",
        })
        writer.writerow({
            "grade": "AISI 316",
            "carbon": "0.03",
            "source": "laboratory",
        })
def test_local_dataset_search_and_provenance(tmp_path, monkeypatch):
    write_csv(tmp_path / "steels.csv")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    result = service.search_local_dataset_records(
        filename="steels.csv", query="laboratory", limit=1
    )
    assert result["total_source_records"] == 3
    assert result["total_matches"] == 2
    assert result["record_count"] == 1
    assert result["records"][0]["grade"] == "AISI 1045"
    assert result["source_provenance"]["classification"] == (
        "LOCAL_DATA_AS_PROVIDED"
    )
    assert len(result["source_provenance"]["source_sha256"]) == 64
def test_local_dataset_pagination(tmp_path, monkeypatch):
    write_csv(tmp_path / "steels.csv")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    result = service.search_local_dataset_records(
        filename="steels.csv", limit=1, offset=1
    )
    assert result["record_count"] == 1
    assert result["records"][0]["grade"] == "AISI 4140"
def test_local_dataset_discovery(tmp_path, monkeypatch):
    write_csv(tmp_path / "steels.csv")
    (tmp_path / "ignore.txt").write_text("not a dataset")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    result = service.list_local_datasets()
    assert result["dataset_count"] == 1
    assert result["datasets"][0]["filename"] == "steels.csv"
@pytest.mark.parametrize("filename", [
    "../outside.csv",
    "subfolder/data.csv",
    "unsupported.json",
    "",
])
def test_local_dataset_rejects_unsafe_or_unsupported_names(
    tmp_path, monkeypatch, filename
):
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    with pytest.raises(ValueError):
        service.search_local_dataset_records(filename=filename)
def test_local_dataset_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        service.search_local_dataset_records(filename="missing.csv")
def test_local_dataset_rejects_duplicate_headers(tmp_path, monkeypatch):
    path = tmp_path / "duplicate.csv"
    path.write_text("grade,grade\nA,B\n", encoding="utf-8")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    with pytest.raises(service.LocalDatasetError, match="duplicate"):
        service.search_local_dataset_records(filename="duplicate.csv")
def test_local_dataset_rejects_extra_csv_fields(tmp_path, monkeypatch):
    path = tmp_path / "extra.csv"
    path.write_text("grade,carbon\nA,0.4,unexpected\n", encoding="utf-8")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    with pytest.raises(service.LocalDatasetError, match="extra fields"):
        service.search_local_dataset_records(filename="extra.csv")
def test_local_dataset_validates_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    with pytest.raises(ValueError):
        service.search_local_dataset_records(filename="x.csv", limit=101)
    with pytest.raises(ValueError):
        service.search_local_dataset_records(filename="x.csv", offset=-1)
def test_local_dataset_api_routes_and_openapi(monkeypatch, tmp_path):
    write_csv(tmp_path / "steels.csv")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    listing = client.get("/api/v1/datasets/local")
    assert listing.status_code == 200
    assert listing.json()["dataset_count"] == 1
    records = client.get(
        "/api/v1/datasets/local/records",
        params={"filename": "steels.csv", "limit": 2},
    )
    assert records.status_code == 200
    assert records.json()["record_count"] == 2
    invalid = client.get(
        "/api/v1/datasets/local/records",
        params={"filename": "../outside.csv"},
    )
    assert invalid.status_code == 422
    openapi = client.get("/openapi.json").json()
    assert "/api/v1/datasets/local" in openapi["paths"]
    assert "/api/v1/datasets/local/records" in openapi["paths"]
def test_local_xlsx_dataset_support(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["grade", "carbon"])
    sheet.append(["AISI 1045", 0.45])
    workbook.save(tmp_path / "steels.xlsx")
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path)
    result = service.search_local_dataset_records(filename="steels.xlsx")
    assert result["format"] == "XLSX"
    assert result["record_count"] == 1
    assert result["records"][0]["grade"] == "AISI 1045"
def test_local_dataset_discovery_falls_back_to_packaged_steelbench(
    monkeypatch, tmp_path
):
    from app.core.paths import PROJECT_ROOT
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path / "legacy")
    result = service.list_local_datasets()
    expected = (
        Path(PROJECT_ROOT)
        / "data"
        / "raw"
        / "steelbench"
        / "SteelBench v1.0.csv"
    )
    assert expected.is_file()
    assert result["dataset_count"] >= 1
    assert any(
        item["filename"] == "SteelBench v1.0.csv"
        for item in result["datasets"]
    )
    assert Path(result["dataset_directory"]).resolve() == expected.parent.resolve()
def test_local_dataset_search_falls_back_to_packaged_steelbench(
    monkeypatch, tmp_path
):
    from app.core.paths import PROJECT_ROOT
    monkeypatch.setattr(service, "DATASET_ROOT", tmp_path / "legacy")
    result = service.search_local_dataset_records(
        filename="SteelBench v1.0.csv",
        limit=2,
    )
    assert result["status"] == "SUCCESS"
    assert result["total_source_records"] == 1360
    assert result["record_count"] == 2
    assert len(result["source_provenance"]["source_sha256"]) == 64

