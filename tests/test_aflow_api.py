import httpx
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
SAMPLE_RECORD = {
    "1 of 3936": {
        "compound": "Ag1Al1Fe2",
        "auid": "aflow:01815824bab78ad1",
        "aurl": "aflowlib.duke.edu:AFLOWDATA/LIB3_WEB/AgAlFe_pv/T0001.ABC2",
        "spacegroup_relax": 225,
        "Pearson_symbol_relax": "cF16",
        "species": ["Ag", "Al", "Fe"],
        "Egap": 0,
        "enthalpy_formation_atom": 0.191026,
    }
}
def fake_response(payload, status_code=200):
    request = httpx.Request(
        "GET",
        "https://aflow.org/API/aflux/",
    )
    return httpx.Response(
        status_code,
        json=payload,
        request=request,
    )
def test_aflow_search_success(monkeypatch):
    captured = {}
    def mock_get(url, timeout, follow_redirects):
        captured["url"] = url
        captured["timeout"] = timeout
        captured["follow_redirects"] = follow_redirects
        return fake_response(SAMPLE_RECORD)
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        mock_get,
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["provider"] == "AFLOW"
    assert data["classification"] == "COMPUTATIONAL_DFT"
    assert data["record_count"] == 1
    assert data["total_matches"] == 3936
    assert captured["follow_redirects"] is False
    record = data["records"][0]
    assert record["auid"] == "aflow:01815824bab78ad1"
    assert record["band_gap"] == 0.0
    assert record["formation_enthalpy"] == 0.191026
    assert record["source"]["data_origin"] == "COMPUTATIONAL"
    assert data["property_units"]["band_gap"] == "eV"
    assert data["property_units"]["formation_enthalpy"] == "eV/atom"
    assert "species(Fe)" in captured["url"]
    assert "Egap(*)" in captured["url"]
    assert "enthalpy_formation_atom(*)" in captured["url"]
def test_aflow_search_rejects_invalid_element(monkeypatch):
    def unexpected_get(*args, **kwargs):
        raise AssertionError("AFLOW must not be called for invalid input.")
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        unexpected_get,
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Xx"},
    )
    assert response.status_code == 422
def test_aflow_search_rejects_missing_elements():
    response = client.get(
        "/api/v1/datasets/aflow/search",
    )
    assert response.status_code == 422
def test_aflow_search_handles_timeout(monkeypatch):
    def mock_get(*args, **kwargs):
        raise httpx.TimeoutException("test timeout")
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        mock_get,
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
    assert "timed out" in response.json()["detail"].lower()
def test_aflow_search_handles_upstream_http_error(monkeypatch):
    def mock_get(*args, **kwargs):
        return fake_response({"error": "upstream failure"}, 500)
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        mock_get,
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
    assert "HTTP 500" in response.json()["detail"]
def test_aflow_search_handles_malformed_record(monkeypatch):
    malformed = {
        "1 of 1": {
            "compound": "Fe",
            "auid": "invalid-id",
            "aurl": "invalid",
            "species": ["Fe"],
        }
    }
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(malformed),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
def test_aflow_rejects_non_numeric_property(monkeypatch):
    malformed = {
        "1 of 1": {
            **SAMPLE_RECORD["1 of 3936"],
            "Egap": "not-a-number",
        }
    }
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(malformed),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
def test_aflow_rejects_missing_required_property(monkeypatch):
    incomplete = {
        "1 of 1": {
            **SAMPLE_RECORD["1 of 3936"],
            "Egap": None,
        }
    }
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(incomplete),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe", "require_properties": "true"},
    )
    assert response.status_code == 503
def test_aflow_rejects_species_mismatch(monkeypatch):
    mismatched = {
        "1 of 1": {
            **SAMPLE_RECORD["1 of 3936"],
            "species": ["Ag", "Al"],
        }
    }
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(mismatched),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
def test_aflow_returns_correct_species_count_and_locator(monkeypatch):
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(SAMPLE_RECORD),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["number_of_species"] == 3
    assert record["source"]["record_locator"] == record["aurl"]
    assert "record_url" not in record["source"]
def test_aflow_accepts_integer_string_space_group(monkeypatch):
    payload = {
        "1 of 1": {
            **SAMPLE_RECORD["1 of 3936"],
            "spacegroup_relax": "225",
        }
    }
    monkeypatch.setattr(
        "app.integrations.aflow_service.httpx.get",
        lambda *args, **kwargs: fake_response(payload),
    )
    response = client.get(
        "/api/v1/datasets/aflow/search",
        params={"elements": "Fe"},
    )
    assert response.status_code == 200
    assert response.json()["records"][0]["space_group"] == 225
def test_aflow_rejects_invalid_space_groups(monkeypatch):
    invalid_values = [225.7, True, 0, 231, "NaN", "not-a-number"]
    for invalid_value in invalid_values:
        payload = {
            "1 of 1": {
                **SAMPLE_RECORD["1 of 3936"],
                "spacegroup_relax": invalid_value,
            }
        }
        monkeypatch.setattr(
            "app.integrations.aflow_service.httpx.get",
            lambda *args, _payload=payload, **kwargs: fake_response(_payload),
        )
        response = client.get(
            "/api/v1/datasets/aflow/search",
            params={"elements": "Fe"},
        )
        assert response.status_code == 503, (
            f"Expected rejection for space group {invalid_value!r}"
        )
        assert "invalid space group" in response.json()["detail"].lower()
