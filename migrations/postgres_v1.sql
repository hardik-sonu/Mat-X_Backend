-- MAT-X PostgreSQL schema candidate
-- Derived from verified local SQLite schema version 4.
-- Generated without modifying the source database.

BEGIN;

-- Table: schema_migrations
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: materials
CREATE TABLE materials (
    material_id TEXT PRIMARY KEY,
    material_family TEXT,
    grade TEXT,
    designation TEXT,
    description TEXT,
    identity_status TEXT NOT NULL DEFAULT 'UNCONFIRMED',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: sources
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    dataset_name TEXT,
    doi_or_url TEXT,
    publication TEXT,
    original_record_id TEXT,
    method TEXT,
    provenance_classification TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: evidence_records
CREATE TABLE evidence_records (
    evidence_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    investigation_id TEXT,
    evidence_category TEXT NOT NULL,
    measured_quantity TEXT NOT NULL,
    value_json TEXT,
    unit TEXT,
    uncertainty_value DOUBLE PRECISION,
    measurement_method TEXT,
    test_standard TEXT,
    source_type TEXT NOT NULL CHECK (
        source_type IN (
            'USER_SUPPLIED',
            'LABORATORY',
            'DATASET',
            'COMPUTATIONAL'
        )
    ),
    source_name TEXT,
    source_record_id TEXT,
    doi_or_url TEXT,
    specimen_id TEXT,
    observed_at TEXT,
    provenance_classification TEXT NOT NULL CHECK (
        provenance_classification IN (
            'SOURCE_VALUE',
            'DERIVED_VALUE',
            'AI_INTERPRETATION',
            'UNAVAILABLE'
        )
    ),
    validation_status TEXT NOT NULL DEFAULT 'UNVERIFIED' CHECK (
        validation_status IN (
            'UNVERIFIED',
            'VALIDATED',
            'REJECTED'
        )
    ),
    original_filename TEXT,
    stored_filename TEXT,
    sha256 TEXT,
    media_type TEXT,
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    ),
    CHECK (
        uncertainty_value IS NULL OR uncertainty_value >= 0
    )
);

-- Table: investigations
CREATE TABLE investigations (
    investigation_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    initial_state_json TEXT NOT NULL DEFAULT '{}',
    final_state_json TEXT,
    limitations_json TEXT NOT NULL DEFAULT '[]',
    started_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    ),
    completed_at TEXT
);

-- Table: material_compositions
CREATE TABLE material_compositions (
    composition_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    element TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL CHECK(value >= 0 AND value <= 100),
    unit TEXT NOT NULL DEFAULT 'wt%',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    ),
    UNIQUE(material_id, element, composition_id)
);

-- Table: process_records
CREATE TABLE process_records (
    process_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    process_type TEXT NOT NULL,
    process_parameters_json TEXT NOT NULL DEFAULT '{}',
    sequence_number INTEGER,
    started_at TEXT,
    completed_at TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: heat_treatments
CREATE TABLE heat_treatments (
    treatment_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    treatment_type TEXT NOT NULL,
    temperature_value DOUBLE PRECISION,
    temperature_unit TEXT,
    holding_time DOUBLE PRECISION,
    holding_time_unit TEXT,
    cooling_method TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: microstructure_records
CREATE TABLE microstructure_records (
    microstructure_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    characterization_method TEXT NOT NULL,
    image_path TEXT,
    map_path TEXT,
    features_json TEXT NOT NULL DEFAULT '{}',
    phases_json TEXT NOT NULL DEFAULT '{}',
    grain_size_value DOUBLE PRECISION,
    grain_size_unit TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: property_records
CREATE TABLE property_records (
    property_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    property_name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    unit TEXT,
    uncertainty_value DOUBLE PRECISION CHECK(
        uncertainty_value IS NULL OR uncertainty_value >= 0
    ),
    measurement_method TEXT,
    test_standard TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: service_records
CREATE TABLE service_records (
    service_record_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    environment_json TEXT NOT NULL DEFAULT '{}',
    temperature_value DOUBLE PRECISION,
    temperature_unit TEXT,
    duration_value DOUBLE PRECISION,
    duration_unit TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: damage_records
CREATE TABLE damage_records (
    damage_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    damage_type TEXT NOT NULL,
    observation_json TEXT NOT NULL,
    severity_value DOUBLE PRECISION,
    severity_unit TEXT,
    diagnosis_status TEXT NOT NULL DEFAULT 'NOT_DIAGNOSED',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
);

-- Table: prediction_records
CREATE TABLE prediction_records (
    prediction_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_id TEXT REFERENCES investigations(investigation_id),
    target_property TEXT NOT NULL,
    predicted_value DOUBLE PRECISION,
    predicted_unit TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_json TEXT NOT NULL,
    applicability_json TEXT NOT NULL DEFAULT '{}',
    provenance_classification TEXT NOT NULL DEFAULT 'PREDICTED_VALUE',
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    )
, prediction_type TEXT NOT NULL DEFAULT 'NUMERIC', prediction_json TEXT NOT NULL DEFAULT '{}');

-- Table: uncertainty_records
CREATE TABLE uncertainty_records (
                uncertainty_id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL REFERENCES materials(material_id),
                investigation_id TEXT REFERENCES investigations(investigation_id),
                prediction_id TEXT REFERENCES prediction_records(prediction_id),
                uncertainty_category TEXT NOT NULL,
                uncertainty_value DOUBLE PRECISION CHECK(
                    uncertainty_value IS NULL OR uncertainty_value >= 0
                ),
                uncertainty_unit TEXT,
                uncertainty_kind TEXT NOT NULL DEFAULT 'UNSPECIFIED',
                lower_bound DOUBLE PRECISION,
                upper_bound DOUBLE PRECISION,
                methodology TEXT,
                calibration_status TEXT NOT NULL DEFAULT 'NOT_CALIBRATED',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (
                    to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
                )
            );

-- Table: experiments
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_id TEXT REFERENCES investigations(investigation_id),
    experiment_type TEXT NOT NULL,
    objective TEXT,
    recommended_measurement TEXT,
    experimental_method TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    status TEXT NOT NULL DEFAULT 'PLANNED',
    provenance_classification TEXT,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
    ),
    completed_at TEXT
);

-- Table: model_versions
CREATE TABLE model_versions (
                model_version_id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                version_label TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL CHECK(
                    length(artifact_sha256) = 64
                ),
                artifact_path TEXT,
                framework TEXT,
                target_property TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                registered_at TEXT NOT NULL DEFAULT (
                    to_char(
                CURRENT_TIMESTAMP AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'
            )
                ),
                UNIQUE(model_key, artifact_sha256)
            );

-- Index: idx_composition_material
CREATE INDEX idx_composition_material
ON material_compositions(material_id);

-- Index: idx_damage_material
CREATE INDEX idx_damage_material
ON damage_records(material_id);

-- Index: idx_evidence_category
CREATE INDEX idx_evidence_category
ON evidence_records(evidence_category);

-- Index: idx_evidence_created
CREATE INDEX idx_evidence_created
ON evidence_records(created_at);

-- Index: idx_evidence_investigation
CREATE INDEX idx_evidence_investigation
ON evidence_records(investigation_id);

-- Index: idx_evidence_material
CREATE INDEX idx_evidence_material
ON evidence_records(material_id);

-- Index: idx_experiment_material
CREATE INDEX idx_experiment_material
ON experiments(material_id);

-- Index: idx_heat_treatment_material
CREATE INDEX idx_heat_treatment_material
ON heat_treatments(material_id);

-- Index: idx_investigation_material
CREATE INDEX idx_investigation_material
ON investigations(material_id);

-- Index: idx_microstructure_material
CREATE INDEX idx_microstructure_material
ON microstructure_records(material_id);

-- Index: idx_model_versions_key
CREATE INDEX idx_model_versions_key
            ON model_versions(model_key);

-- Index: idx_prediction_material
CREATE INDEX idx_prediction_material
ON prediction_records(material_id);

-- Index: idx_process_material
CREATE INDEX idx_process_material
ON process_records(material_id);

-- Index: idx_property_material
CREATE INDEX idx_property_material
ON property_records(material_id);

-- Index: idx_service_material
CREATE INDEX idx_service_material
ON service_records(material_id);

-- Index: idx_uncertainty_material
CREATE INDEX idx_uncertainty_material
            ON uncertainty_records(material_id);

-- Preserve evidence immutability.
CREATE OR REPLACE FUNCTION matx_prevent_evidence_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'Evidence records are immutable.';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Evidence records cannot be deleted.';
    END IF;
    RETURN NULL;
END;
$$;
CREATE TRIGGER evidence_no_update
BEFORE UPDATE ON evidence_records
FOR EACH ROW
EXECUTE FUNCTION matx_prevent_evidence_mutation();
CREATE TRIGGER evidence_no_delete
BEFORE DELETE ON evidence_records
FOR EACH ROW
EXECUTE FUNCTION matx_prevent_evidence_mutation();

-- Preserve the existing schema migration history.
INSERT INTO schema_migrations (version, description, applied_at) VALUES (1, 'Initial MAT-X scientific persistence schema', '2026-11-14T21:30:35.965Z');
INSERT INTO schema_migrations (version, description, applied_at) VALUES (2, 'Uncertainty units and kinds; remove normalized 0-1 restriction', '2026-11-14T22:11:23.895Z');
INSERT INTO schema_migrations (version, description, applied_at) VALUES (3, 'Typed prediction payloads for numeric and classification results', '2026-11-14T22:44:57.366Z');
INSERT INTO schema_migrations (version, description, applied_at) VALUES (4, 'Persistent model-version registry with artifact SHA256 identity', '2026-11-15T02:00:10.515Z');

COMMIT;
