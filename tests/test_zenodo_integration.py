import httpx
import pytest
from fastapi.testclient import TestClient
from app.integrations import zenodo_service as service
SAMPLE = {
    "hits": {
        "total": {"value": 42},
        "hits": [{
            "id": 12345,
            "doi": "10.5281/zenodo.12345",
            "metadata": {
                "title": "Steel microstructure dataset",
                "publication_date": "2025-01-01",
                "description": "Research data for steel characterization.",
                "creators": [{
                    "name": "Researcher A",
                    "affiliation": "Materials Institute",
                    "orcid": "0000-0000-0000-0000",
                }],
                "keywords": ["steel", "microstructure"],
                "resource_type": {"title": "Dataset", "type": "dataset"},
                "license": {"id": "cc-by-4.0"},
            },
            "files": [{
                "key": "steel.csv",
                "size": 2048,
                "checksum": "md5:abc123",
                "links": {"self": "https://zenodo.org/api/records/12345/files/steel.csv"},
            }],
            "links": {
                "html": "https://zenodo.org/records/12345",
                "self": "https://zenodo.org/api/records/12345",
            },
        }],
    }
}
class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", service.ZENODO_API)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "HTTP error", request=request, response=response
            )
    def json(self):
        return self.payload
def test_zenodo_search_normalizes_metadata_and_provenance(monkeypatch):
    captured = {}
    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(SAMPLE)
    monkeypatch.setattr(service.httpx, "get", fake_get)
    result = service.search_zenodo_records(query="steel microstructure")
    assert result["total_matches"] == 42
    assert result["record_count"] == 1
    record = result["records"][0]
    assert record["record_id"] == "12345"
    assert record["doi"] == "10.5281/zenodo.12345"
    assert record["creators"][0]["name"] == "Researcher A"
    assert record["files"][0]["size_bytes"] == 2048
    assert record["source_provenance"]["classification"] == (
        "EXTERNAL_PUBLISHED_REPOSITORY_METADATA"
    )
    assert captured["params"]["size"] == 10
    assert captured["follow_redirects"] is False
    assert captured["headers"]["User-Agent"].startswith("MAT-X/")
def test_zenodo_search_supports_pagination(monkeypatch):
    monkeypatch.setattr(
        service.httpx, "get",
        lambda *args, **kwargs: FakeResponse(SAMPLE),
    )
    result = service.search_zenodo_records(
        query="steel", page=2, size=25, sort="mostrecent"
    )
    assert result["page"] == 2
    assert result["page_size"] == 25
@pytest.mark.parametrize("kwargs", [
    {"query": ""},
    {"query": "steel", "page": 0},
    {"query": "steel", "size": 26},
    {"query": "steel", "sort": "random"},
])
def test_zenodo_rejects_invalid_search_parameters(kwargs):
    with pytest.raises(ValueError):
        service.search_zenodo_records(**kwargs)
def test_zenodo_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(
        service.httpx, "get",
        lambda *args, **kwargs: FakeResponse({"unexpected": []}),
    )
    with pytest.raises(service.ZenodoIntegrationError, match="hits"):
        service.search_zenodo_records(query="steel")
def test_zenodo_handles_http_failure(monkeypatch):
    monkeypatch.setattr(
        service.httpx, "get",
        lambda *args, **kwargs: FakeResponse({}, status_code=429),
    )
    with pytest.raises(service.ZenodoIntegrationError, match="HTTP 429"):
        service.search_zenodo_records(query="steel")
def test_zenodo_handles_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        request = httpx.Request("GET", service.ZENODO_API)
        raise httpx.TimeoutException("timeout", request=request)
    monkeypatch.setattr(service.httpx, "get", timeout)
    with pytest.raises(service.ZenodoIntegrationError, match="timed out"):
        service.search_zenodo_records(query="steel")
def test_zenodo_api_route_and_openapi(monkeypatch):
    monkeypatch.setattr(
        service, "search_zenodo_records",
        lambda **kwargs: {
            "status": "SUCCESS",
            "provider": "Zenodo",
            "query": kwargs["query"],
            "page": kwargs["page"],
            "page_size": kwargs["size"],
            "record_count": 0,
            "total_matches": 0,
            "records": [],
            "source_provenance": {
                "repository": "Zenodo",
                "classification": "EXTERNAL_PUBLISHED_REPOSITORY_METADATA",
                "api_endpoint": service.ZENODO_API,
            },
            "scientific_notice": "Metadata only.",
        },
    )
    from app.main import app
    client = TestClient(app)
    response = client.get(
        "/api/v1/datasets/zenodo/search",
        params={"query": "steel", "size": 5},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "Zenodo"
    invalid = client.get(
        "/api/v1/datasets/zenodo/search",
        params={"query": "steel", "size": 26},
    )
    assert invalid.status_code == 422
    openapi = client.get("/openapi.json").json()
    assert "/api/v1/datasets/zenodo/search" in openapi["paths"]
