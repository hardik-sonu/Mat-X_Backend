import sqlite3
from app.services.evidence_registry_service import (
    initialize_evidence_registry,
    create_evidence_record,
    list_evidence_records,
)
from app.services.matx_database_service import (
    initialize_matx_database,
    SCHEMA_VERSION,
)
def test_expanded_schema_preserves_existing_evidence(tmp_path):
    db = tmp_path / "matx_migration.sqlite3"
    initialize_evidence_registry(db)
    record = create_evidence_record({
        "material_id": "MATX-MIGRATION-001",
        "evidence_category": "mechanical_property",
        "measured_quantity": "hardness",
        "value": 220,
        "unit": "HV",
        "source_type": "LABORATORY",
        "provenance_classification": "SOURCE_VALUE",
    }, db)
    result = initialize_matx_database(db)
    assert result["integrity"] == "ok"
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["foreign_key_errors"] == 0
    assert "evidence_records" in result["tables"]
    assert "materials" in result["tables"]
    assert "investigations" in result["tables"]
    assert "prediction_records" in result["tables"]
    preserved = list_evidence_records(
        material_id="MATX-MIGRATION-001",
        db_path=db,
    )
    assert len(preserved) == 1
    assert preserved[0]["evidence_id"] == record["evidence_id"]
    assert preserved[0]["value"] == 220
def test_expanded_schema_initialization_is_idempotent(tmp_path):
    db = tmp_path / "matx_idempotent.sqlite3"
    initialize_evidence_registry(db)
    first = initialize_matx_database(db)
    second = initialize_matx_database(db)
    assert first["tables"] == second["tables"]
    assert first["schema_version"] == second["schema_version"]
def test_material_and_investigation_relationships(tmp_path):
    db = tmp_path / "matx_relationships.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO materials (material_id, material_family) "
            "VALUES (?, ?)",
            ("MATX-TEST-REL", "Engineering steel"),
        )
        connection.execute(
            """
            INSERT INTO investigations
                (investigation_id, material_id, investigation_goal)
            VALUES (?, ?, ?)
            """,
            ("INV-TEST-REL", "MATX-TEST-REL", "State reconstruction"),
        )
        count = connection.execute(
            "SELECT COUNT(*) FROM investigations "
            "WHERE material_id = ?",
            ("MATX-TEST-REL",),
        ).fetchone()[0]
        assert count == 1
def test_foreign_key_rejects_unknown_material(tmp_path):
    db = tmp_path / "matx_foreign_keys.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.execute(
                """
                INSERT INTO investigations
                    (investigation_id, material_id, investigation_goal)
                VALUES (?, ?, ?)
                """,
                ("INV-INVALID", "UNKNOWN-MATERIAL", "Test"),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Foreign key constraint was not enforced")

def test_expanded_schema_preserves_immutable_evidence_triggers(tmp_path):
    import sqlite3
    db = tmp_path / "matx_triggers.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    with sqlite3.connect(db) as connection:
        trigger_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
    assert "evidence_no_update" in trigger_names
    assert "evidence_no_delete" in trigger_names
def test_all_required_scientific_tables_exist(tmp_path):
    import sqlite3
    db = tmp_path / "matx_tables.sqlite3"
    initialize_evidence_registry(db)
    result = initialize_matx_database(db)
    required = {
        "materials",
        "sources",
        "material_compositions",
        "process_records",
        "heat_treatments",
        "microstructure_records",
        "property_records",
        "service_records",
        "damage_records",
        "investigations",
        "prediction_records",
        "uncertainty_records",
        "experiments",
        "evidence_records",
        "schema_migrations",
    }
    assert required.issubset(set(result["tables"]))
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []

def _create_version_1_database(db):
    """Build an authentic version-1 database for migration testing."""
    from app.services.matx_database_service import SCHEMA
    initialize_evidence_registry(db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            "BEGIN IMMEDIATE;\n"
            + SCHEMA
            + """
            INSERT INTO schema_migrations (version, description)
            VALUES (1, 'Initial MAT-X scientific persistence schema');
            COMMIT;
            """
        )
def test_v2_migration_preserves_uncertainty_records_and_adds_units(tmp_path):
    db = tmp_path / "matx_v1_to_v2.sqlite3"
    _create_version_1_database(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO materials (material_id) VALUES (?)",
            ("UNC-MAT-001",),
        )
        connection.execute(
            """
            INSERT INTO uncertainty_records (
                uncertainty_id, material_id, uncertainty_category,
                uncertainty_value, methodology
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "UNC-001",
                "UNC-MAT-001",
                "LEGACY_UNCERTAINTY",
                0.75,
                "Legacy methodology",
            ),
        )
    result = initialize_matx_database(db)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["integrity"] == "ok"
    assert result["foreign_key_errors"] == 0
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            """
            SELECT uncertainty_id, uncertainty_value,
                   uncertainty_unit, uncertainty_kind, methodology
            FROM uncertainty_records
            WHERE uncertainty_id = ?
            """,
            ("UNC-001",),
        ).fetchone()
        assert row == (
            "UNC-001",
            0.75,
            None,
            "UNSPECIFIED",
            "Legacy methodology",
        )
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert versions == [(1,), (2,), (3,), (4,)]
def test_v2_accepts_absolute_uncertainty_above_one(tmp_path):
    db = tmp_path / "matx_absolute_uncertainty.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO materials (material_id) VALUES (?)",
            ("UNC-MAT-002",),
        )
        connection.execute(
            """
            INSERT INTO uncertainty_records (
                uncertainty_id, material_id, uncertainty_category,
                uncertainty_value, uncertainty_unit, uncertainty_kind
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "UNC-ABS-001",
                "UNC-MAT-002",
                "MECHANICAL_PROPERTY",
                25.0,
                "MPa",
                "ABSOLUTE",
            ),
        )
        row = connection.execute(
            """
            SELECT uncertainty_value, uncertainty_unit, uncertainty_kind
            FROM uncertainty_records WHERE uncertainty_id = ?
            """,
            ("UNC-ABS-001",),
        ).fetchone()
        assert row == (25.0, "MPa", "ABSOLUTE")
def test_v2_rejects_negative_uncertainty(tmp_path):
    db = tmp_path / "matx_negative_uncertainty.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO materials (material_id) VALUES (?)",
            ("UNC-MAT-003",),
        )
        try:
            connection.execute(
                """
                INSERT INTO uncertainty_records (
                    uncertainty_id, material_id, uncertainty_category,
                    uncertainty_value
                ) VALUES (?, ?, ?, ?)
                """,
                ("UNC-NEG-001", "UNC-MAT-003", "TEST", -2.0),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Negative uncertainty was accepted.")
def test_v2_migration_is_idempotent_and_preserves_evidence_triggers(tmp_path):
    db = tmp_path / "matx_v2_idempotent.sqlite3"
    initialize_evidence_registry(db)
    first = initialize_matx_database(db)
    second = initialize_matx_database(db)
    assert first["schema_version"] == SCHEMA_VERSION
    assert second["schema_version"] == SCHEMA_VERSION
    assert first["tables"] == second["tables"]
    with sqlite3.connect(db) as connection:
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert "evidence_no_update" in triggers
        assert "evidence_no_delete" in triggers
        assert connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []
