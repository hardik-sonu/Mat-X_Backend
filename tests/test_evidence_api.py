import json

from fastapi.testclient import TestClient
from app.main import app
from app.services import evidence_registry_service
from app.services import evidence_registry_service as registry
from app.services import evidence_file_service as file_service
client = TestClient(app)
def install_test_database(monkeypatch, tmp_path):
    db_path = tmp_path / "api_evidence.sqlite3"
    registry.initialize_evidence_registry(db_path)
    upload_dir = tmp_path / "uploaded_files"
    upload_dir.mkdir()
    monkeypatch.setattr(
        "app.api.routes.EVIDENCE_FILES_DIR",
        upload_dir,
    )
    async def store_upload(upload_file):
        return await file_service.store_evidence_upload(
            upload_file,
            upload_dir,
        )
    def delete_upload(stored_filename):
        return file_service.delete_stored_evidence_file(
            stored_filename,
            upload_dir,
        )
    monkeypatch.setattr(
        "app.api.routes.store_evidence_upload",
        store_upload,
    )
    monkeypatch.setattr(
        "app.api.routes.delete_stored_evidence_file",
        delete_upload,
    )
    monkeypatch.setattr(
        "app.api.routes.create_evidence_record",
        lambda payload: registry.create_evidence_record(payload, db_path),
    )
    monkeypatch.setattr(
        "app.api.routes.get_evidence_record",
        lambda evidence_id: registry.get_evidence_record(
            evidence_id, db_path
        ),
    )
    monkeypatch.setattr(
        "app.api.routes.list_evidence_records",
        lambda **kwargs: registry.list_evidence_records(
            db_path=db_path, **kwargs
        ),
    )
    return db_path
def valid_payload():
    return {
        "material_id": "MATX-API-001",
        "investigation_id": "INV-API-001",
        "evidence_category": "mechanical_testing",
        "measured_quantity": "hardness",
        "value": 220,
        "unit": "HV",
        "uncertainty_value": 5,
        "measurement_method": "Vickers hardness test",
        "source_type": "LABORATORY",
        "source_name": "MAT-X API Test Lab",
        "provenance_classification": "SOURCE_VALUE",
    }
def test_create_evidence_endpoint(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    response = client.post("/api/v1/evidence", json=valid_payload())
    assert response.status_code == 201
    data = response.json()
    assert data["material_id"] == "MATX-API-001"
    assert data["value"] == 220
    assert data["unit"] == "HV"
    assert data["uncertainty_value"] == 5
    assert data["measurement_method"] == "Vickers hardness test"
    assert data["source_type"] == "LABORATORY"
    assert data["source_name"] == "MAT-X API Test Lab"
    assert data["validation_status"] == "UNVERIFIED"
    assert data["created_at"]
    assert data["evidence_id"]
def test_create_evidence_rejects_invalid_payload(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["source_type"] = "UNKNOWN"
    response = client.post("/api/v1/evidence", json=payload)
    assert response.status_code == 422
def test_retrieve_evidence_endpoint(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    created = client.post(
        "/api/v1/evidence",
        json=valid_payload(),
    ).json()
    response = client.get(
        f"/api/v1/evidence/{created['evidence_id']}"
    )
    assert response.status_code == 200
    retrieved = response.json()
    assert retrieved["evidence_id"] == created["evidence_id"]
    assert retrieved["value"] == 220
    assert retrieved["unit"] == "HV"
    assert retrieved["uncertainty_value"] == 5
    assert retrieved["measurement_method"] == "Vickers hardness test"
def test_retrieve_missing_evidence_returns_404(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    response = client.get(
        "/api/v1/evidence/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
def test_list_evidence_filters_by_material(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    client.post("/api/v1/evidence", json=valid_payload())
    other = valid_payload()
    other["material_id"] = "MATX-API-002"
    client.post("/api/v1/evidence", json=other)
    response = client.get(
        "/api/v1/evidence",
        params={"material_id": "MATX-API-001"},
    )
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1
    assert records[0]["material_id"] == "MATX-API-001"
    assert records[0]["value"] == 220
    assert records[0]["unit"] == "HV"
    assert records[0]["uncertainty_value"] == 5
    assert records[0]["measurement_method"] == "Vickers hardness test"
def test_list_evidence_rejects_invalid_limit():
    response = client.get(
        "/api/v1/evidence",
        params={"limit": 1001},
    )
    assert response.status_code == 422
def test_create_evidence_rejects_unknown_fields(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["unexpected_field"] = "not allowed"
    response = client.post("/api/v1/evidence", json=payload)
    assert response.status_code == 422
def test_create_evidence_rejects_boolean_uncertainty(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["uncertainty_value"] = True
    response = client.post("/api/v1/evidence", json=payload)
    assert response.status_code == 422

def test_upload_evidence_endpoint(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    content = b"material_id,hardness\\nMATX-API-001,220\\n"
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={"file": ("hardness.csv", content, "text/csv")},
    )
    assert response.status_code == 201
    record = response.json()
    assert record["material_id"] == "MATX-API-001"
    assert record["value"] == 220
    assert record["unit"] == "HV"
    assert record["uncertainty_value"] == 5
    assert record["measurement_method"] == "Vickers hardness test"
    assert record["original_filename"] == "hardness.csv"
    assert record["stored_filename"].endswith(".csv")
    assert record["media_type"] == "text/csv"
    assert record["sha256"]
    stored_path = (
        tmp_path / "uploaded_files" / record["stored_filename"]
    )
    assert stored_path.exists()
    assert stored_path.read_bytes() == content
def test_upload_rejects_invalid_json_metadata(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": "{invalid json"},
        files={"file": ("hardness.csv", b"a,b\\n1,2\\n", "text/csv")},
    )
    assert response.status_code == 422
    assert list((tmp_path / "uploaded_files").iterdir()) == []
def test_upload_rejects_client_supplied_file_metadata(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["sha256"] = "client-controlled-checksum"
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(payload)},
        files={"file": ("hardness.csv", b"a,b\\n1,2\\n", "text/csv")},
    )
    assert response.status_code == 422
    assert list((tmp_path / "uploaded_files").iterdir()) == []
def test_upload_rejects_mismatched_file_content(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={"file": ("report.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 422
    assert list((tmp_path / "uploaded_files").iterdir()) == []
def test_upload_rolls_back_file_when_database_registration_fails(
    monkeypatch,
    tmp_path,
):
    install_test_database(monkeypatch, tmp_path)
    def fail_database_registration(payload):
        raise RuntimeError("Simulated database failure.")
    monkeypatch.setattr(
        "app.api.routes.create_evidence_record",
        fail_database_registration,
    )
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={
            "file": (
                "hardness.csv",
                b"material_id,hardness\\nMATX-API-001,220\\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Evidence file could not be registered."
    )
    assert list((tmp_path / "uploaded_files").iterdir()) == []

def test_download_uploaded_evidence_file(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    content = b"material_id,hardness\\nMATX-API-001,220\\n"
    uploaded = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={"file": ("hardness.csv", content, "text/csv")},
    )
    assert uploaded.status_code == 201
    evidence_id = uploaded.json()["evidence_id"]
    response = client.get(
        f"/api/v1/evidence/{evidence_id}/file"
    )
    assert response.status_code == 200
    assert response.content == content
    assert "hardness.csv" in response.headers["content-disposition"]
def test_download_missing_evidence_file_returns_404(
    monkeypatch,
    tmp_path,
):
    install_test_database(monkeypatch, tmp_path)
    created = client.post(
        "/api/v1/evidence",
        json=valid_payload(),
    ).json()
    response = client.get(
        f"/api/v1/evidence/{created['evidence_id']}/file"
    )
    assert response.status_code == 404
def test_download_rejects_checksum_mismatch(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    content = b"material_id,hardness\\nMATX-API-001,220\\n"
    uploaded = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={"file": ("hardness.csv", content, "text/csv")},
    )
    assert uploaded.status_code == 201
    record = uploaded.json()
    stored_path = (
        tmp_path / "uploaded_files" / record["stored_filename"]
    )
    stored_path.write_bytes(b"tampered evidence content")
    response = client.get(
        f"/api/v1/evidence/{record['evidence_id']}/file"
    )
    assert response.status_code == 409
    assert "checksum" in response.json()["detail"].lower()
def test_create_evidence_rejects_client_validation_status(
    monkeypatch, tmp_path
):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["validation_status"] = "VALIDATED"
    response = client.post("/api/v1/evidence", json=payload)
    assert response.status_code == 422
    assert "validation_status" in response.text
def test_upload_evidence_rejects_client_validation_status(
    monkeypatch, tmp_path
):
    install_test_database(monkeypatch, tmp_path)
    payload = valid_payload()
    payload["validation_status"] = "VALIDATED"
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(payload)},
        files={"file": ("hardness.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "validation_status" in response.text
    assert list((tmp_path / "uploaded_files").iterdir()) == []
def test_trusted_validated_evidence_contributes_to_material_state(
    monkeypatch, tmp_path
):
    from app.api.routes import MaterialStateReconstructionInput
    import app.services.evidence_registry_service as registry
    import app.services.material_state_service as state_service
    db_path = tmp_path / "validated_evidence.sqlite3"
    registry.initialize_evidence_registry(db_path)
    registry.create_evidence_record(
        {
            "material_id": "MATX-TRUSTED-VALIDATION-TEST",
            "evidence_category": "mechanical_property",
            "measured_quantity": "hardness",
            "value": 220,
            "unit": "HV",
            "source_type": "LABORATORY",
            "provenance_classification": "SOURCE_VALUE",
            "measurement_method": "Vickers hardness test",
            "validation_status": "VALIDATED",
        },
        db_path=db_path,
    )
    monkeypatch.setattr(
        state_service,
        "list_evidence_records",
        lambda material_id=None, investigation_id=None, limit=100:
            registry.list_evidence_records(
                material_id=material_id,
                investigation_id=investigation_id,
                limit=limit,
                db_path=db_path,
            ),
    )
    request = MaterialStateReconstructionInput(
        material_id="MATX-TRUSTED-VALIDATION-TEST",
        material_family="low_alloy_steel",
    )
    state = state_service.reconstruct_material_state(request)
    mechanical = state["evidence"]["mechanical_properties"]
    assert state["evidence_registry"]["validated_usable_record_count"] == 1
    assert mechanical["status"] == "AVAILABLE"
    assert mechanical["source"] == "VALIDATED_EVIDENCE_REGISTRY"
    assert mechanical["validated_registry_record_count"] == 1
    assert mechanical["value"][0]["value"] == 220
    assert mechanical["value"][0]["unit"] == "HV"
def test_end_to_end_evidence_state_investigation_report_workflow(
    monkeypatch, tmp_path
):
    import app.services.material_state_service as state_service
    db_path = install_test_database(monkeypatch, tmp_path)
    material_id = "MATX-E2E-WORKFLOW-001"
    investigation_id = "INV-E2E-WORKFLOW-001"
    monkeypatch.setattr(
        state_service,
        "list_evidence_records",
        lambda material_id=None, investigation_id=None, limit=100:
            registry.list_evidence_records(
                material_id=material_id,
                investigation_id=investigation_id,
                limit=limit,
                db_path=db_path,
            ),
    )
    # 1. Submit evidence through the public API.
    payload = valid_payload()
    payload.update({
        "material_id": material_id,
        "investigation_id": investigation_id,
        "value": 220,
        "unit": "HV",
    })
    created = client.post("/api/v1/evidence", json=payload)
    assert created.status_code == 201
    unverified_record = created.json()
    assert unverified_record["validation_status"] == "UNVERIFIED"
    # 2. Retrieve the persisted record and verify its provenance.
    evidence_id = unverified_record["evidence_id"]
    retrieved = client.get(f"/api/v1/evidence/{evidence_id}")
    assert retrieved.status_code == 200
    assert retrieved.json()["value"] == 220
    assert retrieved.json()["validation_status"] == "UNVERIFIED"
    listed = client.get(
        "/api/v1/evidence",
        params={"material_id": material_id},
    )
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    # 3. Reconstruct state before trusted validation.
    state_request = {
        "material_id": material_id,
        "material_family": "low_alloy_steel",
    }
    first_state = client.post(
        "/api/v1/material/state/reconstruct",
        json=state_request,
    )
    assert first_state.status_code == 200
    first_data = first_state.json()
    assert first_data["evidence_registry"]["record_count"] == 1
    assert (
        first_data["evidence_registry"]["validated_usable_record_count"]
        == 0
    )
    assert (
        first_data["evidence"]["mechanical_properties"]["status"]
        == "UNAVAILABLE"
    )
    assert (
        "mechanical_properties"
        in first_data["active_investigation"]["evidence_summary"][
            "missing_categories"
        ]
    )
    # 4. Simulate trusted internal validation of a new measurement.
    validated_record = registry.create_evidence_record(
        {
            "material_id": material_id,
            "investigation_id": investigation_id,
            "evidence_category": "mechanical_property",
            "measured_quantity": "hardness",
            "value": 225,
            "unit": "HV",
            "source_type": "LABORATORY",
            "source_name": "MAT-X Trusted Validation Lab",
            "provenance_classification": "SOURCE_VALUE",
            "measurement_method": "Vickers hardness test",
            "validation_status": "VALIDATED",
        },
        db_path=db_path,
    )
    assert validated_record["validation_status"] == "VALIDATED"
    # 5. Reconstruct state and verify the accepted evidence changes coverage.
    updated_state = client.post(
        "/api/v1/material/state/reconstruct",
        json=state_request,
    )
    assert updated_state.status_code == 200
    updated_data = updated_state.json()
    assert updated_data["evidence_registry"]["record_count"] == 2
    assert (
        updated_data["evidence_registry"]["validated_usable_record_count"]
        == 1
    )
    mechanical = updated_data["evidence"]["mechanical_properties"]
    assert mechanical["status"] == "AVAILABLE"
    assert mechanical["source"] == "VALIDATED_EVIDENCE_REGISTRY"
    assert mechanical["value"][0]["value"] == 225
    assert mechanical["value"][0]["unit"] == "HV"
    assert (
        "mechanical_properties"
        not in updated_data["active_investigation"]["evidence_summary"][
            "missing_categories"
        ]
    )
    assert (
        updated_data["active_investigation"]["evidence_summary"][
            "available_categories"
        ]
        > first_data["active_investigation"]["evidence_summary"][
            "available_categories"
        ]
    )
    # 6. Exercise the dedicated active-investigation API endpoint.
    investigation_response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": material_id,
            "material_family": "low_alloy_steel",
            "evidence": {
                "mechanical_properties": {
                    "hardness": 225,
                    "unit": "HV",
                }
            },
            "uncertainty": {"mechanical_testing": 0.2},
            "investigation_goal": "material_state_reconstruction",
            "top_k": 7,
        },
    )
    assert investigation_response.status_code == 200, investigation_response.text
    investigation_data = investigation_response.json()
    assert investigation_data["material_id"] == material_id
    assert isinstance(investigation_data["recommendations"], list)
    assert investigation_data["recommendations"]
    # 6. Generate the engineering report from the reconstructed evidence.
    report_response = client.post(
        "/api/v1/engineering/report",
        json=state_request,
    )
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["material"]["material_id"] == material_id
    assert report["report"]["report_status"] == (
        "EVIDENCE_AGGREGATED_NOT_CERTIFIED"
    )
    assert report["validated_evidence_registry"]["record_count"] == 2
    assert (
        report["validated_evidence_registry"]["validated_usable_record_count"]
        == 1
    )
    assert report["prediction_results"]["status"] == "NO_PERSISTED_PREDICTIONS"
    assert report["limitations"]


def test_evidence_response_openapi_documents_scientific_fields():
    from app.main import app
    schema = app.openapi()["components"]["schemas"]["EvidenceRecordResponse"]
    properties = schema["properties"]
    expected_fields = {
        "value",
        "unit",
        "uncertainty_value",
        "measurement_method",
        "test_standard",
        "source_type",
        "source_name",
        "source_record_id",
        "doi_or_url",
        "specimen_id",
        "investigation_id",
        "observed_at",
        "provenance_classification",
        "validation_status",
        "original_filename",
        "stored_filename",
        "sha256",
        "media_type",
        "created_at",
    }
    assert expected_fields.issubset(properties)
    assert properties["uncertainty_value"]["anyOf"][0]["type"] in {
        "number",
        "integer",
    } or properties["uncertainty_value"]["type"] in {
        "number",
        "integer",
    }

def test_active_investigation_api_rejects_unsupported_goal():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-API-GOAL-INVALID",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "investigation_goal": "failure_diagnosis",
        },
    )
    assert response.status_code == 422
def test_upload_xlsx_evidence_endpoint(monkeypatch, tmp_path):
    import hashlib
    import io
    import zipfile
    install_test_database(monkeypatch, tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"/>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheets/></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"/>',
        )
    content = buffer.getvalue()
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={
            "file": (
                "hardness_measurements.xlsx",
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["material_id"] == "MATX-API-001"
    assert record["original_filename"] == "hardness_measurements.xlsx"
    assert record["stored_filename"].endswith(".xlsx")
    assert record["media_type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert record["sha256"] == hashlib.sha256(content).hexdigest()
    stored_path = tmp_path / "uploaded_files" / record["stored_filename"]
    assert stored_path.exists()
    assert stored_path.read_bytes() == content
    downloaded = client.get(
        f"/api/v1/evidence/{record['evidence_id']}/file"
    )
    assert downloaded.status_code == 200
    assert downloaded.content == content
def _build_test_xlsx_bytes():
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"/>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheets/></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"/>',
        )
    return buffer.getvalue()
def test_upload_xlsx_api_rejects_malformed_workbook(monkeypatch, tmp_path):
    install_test_database(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={
            "file": (
                "malformed.xlsx",
                b"this is not a valid XLSX archive",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 422
    assert list((tmp_path / "uploaded_files").iterdir()) == []
def test_upload_xlsx_api_rolls_back_on_database_failure(
    monkeypatch, tmp_path
):
    install_test_database(monkeypatch, tmp_path)
    def fail_database_registration(payload):
        raise RuntimeError("Simulated database failure.")
    monkeypatch.setattr(
        "app.api.routes.create_evidence_record",
        fail_database_registration,
    )
    response = client.post(
        "/api/v1/evidence/upload",
        data={"metadata": json.dumps(valid_payload())},
        files={
            "file": (
                "hardness_measurements.xlsx",
                _build_test_xlsx_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Evidence file could not be registered."
    )
    assert list((tmp_path / "uploaded_files").iterdir()) == []

