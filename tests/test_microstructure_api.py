
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_microstructure_maps_service_failure_returns_500(monkeypatch):
    def fail():
        raise RuntimeError("Simulated service failure.")
    monkeypatch.setattr("app.api.routes.list_microstructure_maps", fail)
    response = client.get("/api/v1/microstructure/maps")
    assert response.status_code == 500
def test_microstructure_features_missing_map_returns_404(monkeypatch):
    def missing(map_id):
        raise FileNotFoundError("EBSD map not found.")
    monkeypatch.setattr("app.api.routes.get_microstructure_features", missing)
    response = client.get("/api/v1/microstructure/maps/unknown/features")
    assert response.status_code == 404
def test_microstructure_features_invalid_map_returns_422(monkeypatch):
    def invalid(map_id):
        raise ValueError("Invalid EBSD map structure.")
    monkeypatch.setattr("app.api.routes.get_microstructure_features", invalid)
    response = client.get("/api/v1/microstructure/maps/invalid/features")
    assert response.status_code == 422
def test_microstructure_provenance_missing_dataset_returns_503(monkeypatch):
    def missing():
        raise FileNotFoundError("EBSD provenance dataset unavailable.")
    monkeypatch.setattr("app.api.routes.get_ebsd_provenance", missing)
    response = client.get("/api/v1/microstructure/provenance")
    assert response.status_code == 503
def test_unified_microstructure_missing_map_returns_404(monkeypatch):
    def missing(map_id):
        raise FileNotFoundError("Unified EBSD map not found.")
    monkeypatch.setattr(
        "app.api.routes.get_unified_microstructure_record", missing
    )
    response = client.get("/api/v1/microstructure/maps/unknown/unified")
    assert response.status_code == 404
def test_unified_microstructure_invalid_map_returns_422(monkeypatch):
    def invalid(map_id):
        raise ValueError("Unified EBSD record is invalid.")
    monkeypatch.setattr(
        "app.api.routes.get_unified_microstructure_record", invalid
    )
    response = client.get("/api/v1/microstructure/maps/invalid/unified")
    assert response.status_code == 422

def test_microstructure_features_success_returns_feature_payload(monkeypatch):
    expected = {
        "map_id": "sample-map",
        "total_points": 100,
        "indexed_points": 92,
        "indexing_success_fraction": 0.92,
        "phase_fractions_all_points": {},
        "phase_fractions_indexed_points": {},
        "provenance": "SOURCE_DATA_AND_DERIVED_FEATURES",
    }
    def success(map_id):
        assert map_id == "sample-map"
        return expected
    monkeypatch.setattr(
        "app.api.routes.get_microstructure_features", success
    )
    response = client.get(
        "/api/v1/microstructure/maps/sample-map/features"
    )
    assert response.status_code == 200
    assert response.json() == expected
