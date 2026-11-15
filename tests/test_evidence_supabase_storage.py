import asyncio
import hashlib
from io import BytesIO
import httpx
import pytest
from starlette.datastructures import UploadFile
from app.services import evidence_file_service as service
class FakeResponse:
    status_code = 200
    def __init__(self, content=b""):
        self.content = content
    def raise_for_status(self):
        return None
class FakeAsyncClient:
    calls = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def post(self, url, **kwargs):
        self.__class__.calls.append((url, kwargs))
        return FakeResponse()
class FakeSyncClient:
    calls = []
    download_content = b""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def get(self, url, **kwargs):
        self.__class__.calls.append(("GET", url, kwargs))
        return FakeResponse(self.__class__.download_content)
    def request(self, method, url, **kwargs):
        self.__class__.calls.append((method, url, kwargs))
        return FakeResponse()
def configure_remote(monkeypatch):
    monkeypatch.setattr(service, "database_backend", lambda: "postgres")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "matx-evidence")
    FakeAsyncClient.calls.clear()
    FakeSyncClient.calls.clear()
def test_remote_upload_uses_private_storage_and_returns_checksum(
    monkeypatch, tmp_path
):
    configure_remote(monkeypatch)
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(service.httpx, "AsyncClient", FakeAsyncClient)
    content = b"material_id,hardness\nMATX-001,220\n"
    upload = UploadFile(filename="hardness.csv", file=BytesIO(content))
    result = asyncio.run(service.store_evidence_upload(upload))
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["size_bytes"] == len(content)
    assert result["storage_path"].startswith("supabase://matx-evidence/")
    assert len(FakeAsyncClient.calls) == 1
    url, request = FakeAsyncClient.calls[0]
    assert "/storage/v1/object/matx-evidence/" in url
    assert request["content"] == content
    assert request["headers"]["x-upsert"] == "false"
    assert request["headers"]["Authorization"] == "Bearer test-secret-key"
    assert list((tmp_path / "MATX_Evidence_Uploads").iterdir()) == []
def test_remote_download_uses_authenticated_private_endpoint(monkeypatch):
    configure_remote(monkeypatch)
    content = b"verified evidence content"
    FakeSyncClient.download_content = content
    monkeypatch.setattr(service.httpx, "Client", FakeSyncClient)
    result = service.download_stored_evidence_file("evidence123.txt")
    assert result == content
    method, url, request = FakeSyncClient.calls[-1]
    assert method == "GET"
    assert "/storage/v1/object/authenticated/matx-evidence/" in url
    assert request["headers"]["apikey"] == "test-secret-key"
def test_remote_delete_uses_storage_remove_endpoint(monkeypatch):
    configure_remote(monkeypatch)
    monkeypatch.setattr(service.httpx, "Client", FakeSyncClient)
    service.delete_stored_evidence_file("evidence123.txt")
    method, url, request = FakeSyncClient.calls[-1]
    assert method == "DELETE"
    assert url.endswith("/storage/v1/object/matx-evidence")
    assert request["json"] == {"prefixes": ["evidence123.txt"]}
def test_remote_upload_rejects_missing_supabase_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(service, "database_backend", lambda: "postgres")
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_STORAGE_BUCKET", raising=False)
    upload = UploadFile(
        filename="hardness.csv",
        file=BytesIO(b"material_id,hardness\nMATX-001,220\n"),
    )
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        asyncio.run(service.store_evidence_upload(upload))
    assert not (tmp_path / "MATX_Evidence_Uploads").exists()
def test_remote_upload_failure_cleans_local_staging(monkeypatch, tmp_path):
    configure_remote(monkeypatch)
    monkeypatch.setattr(service.tempfile, "gettempdir", lambda: str(tmp_path))
    class FailingAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            raise httpx.ConnectError("mocked connection failure")
    monkeypatch.setattr(service.httpx, "AsyncClient", FailingAsyncClient)
    upload = UploadFile(
        filename="hardness.csv",
        file=BytesIO(b"material_id,hardness\nMATX-001,220\n"),
    )
    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(service.store_evidence_upload(upload))
    assert list((tmp_path / "MATX_Evidence_Uploads").iterdir()) == []
