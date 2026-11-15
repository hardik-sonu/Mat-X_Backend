from fastapi.testclient import TestClient
import hashlib
from app.main import app
from app.api import routes
client = TestClient(app)
def install_remote_record(monkeypatch, content):
    record = {
        "evidence_id": "REMOTE-EVIDENCE-001",
        "stored_filename": "evidence123.txt",
        "original_filename": "hardness-report.txt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": "text/plain",
    }
    monkeypatch.setattr(
        routes,
        "get_evidence_record",
        lambda evidence_id: record if evidence_id == record["evidence_id"] else None,
    )
    monkeypatch.setattr(routes, "is_remote_evidence_storage", lambda: True)
    monkeypatch.setattr(
        routes,
        "download_stored_evidence_file",
        lambda filename: content,
    )
    return record
def test_remote_download_returns_verified_file(monkeypatch):
    content = b"verified remote evidence"
    install_remote_record(monkeypatch, content)
    response = client.get("/api/v1/evidence/REMOTE-EVIDENCE-001/file")
    assert response.status_code == 200
    assert response.content == content
    assert "hardness-report.txt" in response.headers["content-disposition"]
    assert response.headers["content-type"].startswith("text/plain")
def test_remote_download_rejects_checksum_mismatch(monkeypatch):
    content = b"tampered remote evidence"
    record = install_remote_record(monkeypatch, content)
    record["sha256"] = hashlib.sha256(b"different content").hexdigest()
    response = client.get("/api/v1/evidence/REMOTE-EVIDENCE-001/file")
    assert response.status_code == 409
    assert "checksum" in response.json()["detail"].lower()
def test_remote_download_returns_404_when_storage_file_missing(monkeypatch):
    install_remote_record(monkeypatch, b"expected evidence")
    def missing_file(filename):
        raise FileNotFoundError(filename)
    monkeypatch.setattr(routes, "download_stored_evidence_file", missing_file)
    response = client.get("/api/v1/evidence/REMOTE-EVIDENCE-001/file")
    assert response.status_code == 404
def test_remote_download_returns_502_when_storage_fails(monkeypatch):
    install_remote_record(monkeypatch, b"expected evidence")
    def storage_failure(filename):
        raise RuntimeError("simulated private storage failure")
    monkeypatch.setattr(
        routes,
        "download_stored_evidence_file",
        storage_failure,
    )
    response = client.get("/api/v1/evidence/REMOTE-EVIDENCE-001/file")
    assert response.status_code == 502
    assert "storage service" in response.json()["detail"].lower()
def test_remote_download_rejects_windows_path_traversal(monkeypatch):
    content = b"evidence"
    record = install_remote_record(monkeypatch, content)
    record["stored_filename"] = r"..\\outside.txt"
    def unexpected_download(filename):
        raise AssertionError("Storage must not be accessed for an invalid path.")
    monkeypatch.setattr(
        routes,
        "download_stored_evidence_file",
        unexpected_download,
    )
    response = client.get("/api/v1/evidence/REMOTE-EVIDENCE-001/file")
    assert response.status_code == 404
def test_remote_download_returns_404_for_unknown_record(monkeypatch):
    monkeypatch.setattr(routes, "get_evidence_record", lambda evidence_id: None)
    monkeypatch.setattr(routes, "is_remote_evidence_storage", lambda: True)
    response = client.get("/api/v1/evidence/UNKNOWN-EVIDENCE/file")
    assert response.status_code == 404
