import httpx
from fastapi.testclient import TestClient
from app.main import app
from app.integrations import jarvis_service
client = TestClient(app)
def fake_jarvis_payload():
    return {
        "meta": {
            "provider": {"name": "NIST-JARVIS"},
            "more_data_available": False,
        },
        "data": [
            {
                "id": "dft_3d_JVASP-104984",
                "attributes": {
                    "_jarvis_jid": "JVASP-104984",
                    "chemical_formula_reduced": "Fe",
                    "elements": ["Fe"],
                    "_jarvis_density": 8.289,
                    "_jarvis_spg_symbol": "Im-3m",
                    "_jarvis_formation_energy": -0.00363,
                    "_jarvis_optb88vdw_bandgap": 0,
                    "_jarvis_reference": "JARVIS-DFT",
                },
            },
            {
                "id": "dft_3d_JVASP-999999",
                "attributes": {
                    "_jarvis_jid": "JVASP-999999",
                    "chemical_formula_reduced": "Fe",
                    "elements": ["Fe"],
                    "_jarvis_density": -99999,
                    "_jarvis_spg_symbol": "Im-3m",
                    "_jarvis_formation_energy": -99999,
                    "_jarvis_optb88vdw_bandgap": -99999,
                    "_jarvis_reference": None,
                },
            },
        ],
    }
class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload
def test_jarvis_endpoint_success(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(fake_jarvis_payload())
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 10},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "NIST-JARVIS"
    assert data["dataset"] == "JARVIS-DFT"
    assert data["classification"] == "COMPUTATIONAL_DFT"
    assert data["record_count"] == 2
    first = data["records"][0]
    assert first["jarvis_id"] == "JVASP-104984"
    assert first["density"] == 8.289
    assert first["band_gap"] == 0
    assert first["source"]["data_origin"] == "COMPUTATIONAL"
    second = data["records"][1]
    assert second["density"] is None
    assert second["formation_energy"] is None
    assert second["band_gap"] is None
def test_jarvis_rejects_invalid_element():
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "fe"},
    )
    assert response.status_code == 422
def test_jarvis_requires_elements():
    response = client.get(
        "/api/v1/datasets/jarvis/structures"
    )
    assert response.status_code == 422
def test_jarvis_upstream_timeout_returns_503(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.TimeoutException("simulated timeout")
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
    assert "timed out" in response.json()["detail"].lower()
def test_jarvis_rejects_malformed_provider_record(monkeypatch):
    malformed_payload = {
        "meta": {
            "provider": {"name": "NIST-JARVIS"},
            "more_data_available": False,
        },
        "data": [{"id": "bad-record", "attributes": {}}],
    }
    def fake_get(*args, **kwargs):
        return FakeResponse(malformed_payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
def test_jarvis_enforces_page_limit_when_provider_over_returns(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(fake_jarvis_payload())
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 1
    assert len(data["records"]) == 1
    assert data["more_data_available"] is True
def test_jarvis_rejects_unknown_periodic_element():
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Xx"},
    )
    assert response.status_code == 422
def test_jarvis_rejects_non_finite_numeric_property(monkeypatch):
    import math
    payload = fake_jarvis_payload()
    payload["data"][0]["attributes"]["_jarvis_density"] = math.nan
    def fake_get(*args, **kwargs):
        return FakeResponse(payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
    assert "numeric" in response.json()["detail"].lower()
def test_jarvis_rejects_malformed_provider_metadata(monkeypatch):
    payload = fake_jarvis_payload()
    payload["meta"]["provider"] = "invalid-provider-object"
    def fake_get(*args, **kwargs):
        return FakeResponse(payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe"},
    )
    assert response.status_code == 503
def test_jarvis_fetches_next_page_and_combines_records(monkeypatch):
    first_payload = fake_jarvis_payload()
    first_payload["data"] = first_payload["data"][:1]
    first_payload["links"] = {
        "next": (
            "https://jarvis.nist.gov/optimade/"
            "jarvisdft/v1/structures/?page_offset=1"
        )
    }
    second_payload = fake_jarvis_payload()
    second_payload["data"] = second_payload["data"][1:]
    calls = []
    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            return FakeResponse(first_payload)
        if len(calls) == 2:
            return FakeResponse(second_payload)
        raise AssertionError("Unexpected extra pagination request.")
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 2
    assert [
        record["jarvis_id"] for record in data["records"]
    ] == ["JVASP-104984", "JVASP-999999"]
    assert len(calls) == 2
    assert calls[1][0].endswith("page_offset=1")
    assert data["more_data_available"] is False
def test_jarvis_stops_pagination_at_total_record_limit(monkeypatch):
    first_payload = fake_jarvis_payload()
    first_payload["data"] = first_payload["data"][:1]
    first_payload["links"] = {
        "next": (
            "https://jarvis.nist.gov/optimade/"
            "jarvisdft/v1/structures/?page_offset=1"
        )
    }
    second_payload = fake_jarvis_payload()
    second_payload["data"] = second_payload["data"][1:]
    calls = []
    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return FakeResponse(first_payload)
        return FakeResponse(second_payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 2},
    )
    assert response.status_code == 200
    assert response.json()["record_count"] == 2
    assert len(calls) == 2
def test_jarvis_rejects_external_pagination_url(monkeypatch):
    payload = fake_jarvis_payload()
    payload["data"] = payload["data"][:1]
    payload["links"] = {
        "next": "https://example.com/steal-data"
    }
    def fake_get(*args, **kwargs):
        return FakeResponse(payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 2},
    )
    assert response.status_code == 503
    assert "unsafe pagination url" in response.json()["detail"].lower()
def test_jarvis_rejects_repeated_pagination_link(monkeypatch):
    payload = fake_jarvis_payload()
    payload["data"] = payload["data"][:1]
    payload["links"] = {"next": jarvis_service.JARVIS_ENDPOINT}
    def fake_get(*args, **kwargs):
        return FakeResponse(payload)
    monkeypatch.setattr(jarvis_service.httpx, "get", fake_get)
    response = client.get(
        "/api/v1/datasets/jarvis/structures",
        params={"elements": "Fe", "page_limit": 2},
    )
    assert response.status_code == 503
    assert "repeated pagination link" in response.json()["detail"].lower()

