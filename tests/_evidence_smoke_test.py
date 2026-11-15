import sqlite3
import tempfile
from pathlib import Path
from app.services.evidence_registry_service import (
    initialize_evidence_registry,
    create_evidence_record,
    get_evidence_record,
    list_evidence_records,
)
db_path = Path(tempfile.gettempdir()) / "matx_evidence_smoke_test.sqlite3"
db_path.unlink(missing_ok=True)
try:
    initialize_evidence_registry(db_path)
    payload = {
        "material_id": "MATX-TEST-001",
        "investigation_id": "INV-TEST-001",
        "evidence_category": "mechanical_testing",
        "measured_quantity": "hardness",
        "value": 220,
        "unit": "HV",
        "uncertainty_value": 5,
        "measurement_method": "Vickers hardness test",
        "source_type": "LABORATORY",
        "source_name": "Test Lab",
        "provenance_classification": "SOURCE_VALUE",
    }
    record = create_evidence_record(payload, db_path)
    assert record["value"] == 220
    assert record["unit"] == "HV"
    retrieved = get_evidence_record(record["evidence_id"], db_path)
    assert retrieved["material_id"] == "MATX-TEST-001"
    records = list_evidence_records(
        material_id="MATX-TEST-001",
        db_path=db_path,
    )
    assert len(records) == 1
    connection = sqlite3.connect(db_path)
    delete_blocked = False
    try:
        connection.execute(
            "DELETE FROM evidence_records WHERE evidence_id = ?",
            (record["evidence_id"],),
        )
        connection.commit()
    except sqlite3.IntegrityError:
        delete_blocked = True
    finally:
        connection.close()
    assert delete_blocked
    print("PASS: evidence insertion")
    print("PASS: evidence retrieval")
    print("PASS: material filtering")
    print("PASS: JSON value round-trip")
    print("PASS: immutable delete protection")
    print("Evidence ID:", record["evidence_id"])
finally:
    db_path.unlink(missing_ok=True)
    Path(str(db_path) + "-journal").unlink(missing_ok=True)
