# MAT-X Database and Data Model

> Generated on 2026-11-15T07:38:34+05:00. Database inspection reads schema metadata only; it does not export application records.

## Logical entities from the Version 1.0 design

| Entity | Purpose |
|---|---|
| materials | Material identity and investigation-level records |
| material_compositions | Chemical composition evidence |
| process_records | Manufacturing and process history |
| heat_treatments | Heat-treatment conditions and history |
| microstructure_records | Microstructure analysis and associated metadata |
| property_records | Observed or recorded material properties |
| service_records | Service and operating conditions |
| damage_records | Damage-related observations |
| prediction_records | Model-generated predictions |
| uncertainty_records | Uncertainty and applicability information |
| investigations | Investigation and recommendation records |
| sources | Dataset and source metadata |
| experiments | Experiment or evaluation records |
| model_versions | Registered model version metadata |
| evidence registry | Uploaded evidence metadata and file references |

The logical design in the specification and the implemented physical schema are not assumed to be identical. The following section reports tables actually found in local SQLite databases.

## Implemented SQLite schema inventory

### `C:/Users/Ahmadcomputer/MATX_Storage/matx_evidence.sqlite3`

#### `damage_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `damage_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `damage_type` | `TEXT` | False | False |
| `observation_json` | `TEXT` | False | False |
| `severity_value` | `REAL` | True | False |
| `severity_unit` | `TEXT` | True | False |
| `diagnosis_status` | `TEXT` | False | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `evidence_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `evidence_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `investigation_id` | `TEXT` | True | False |
| `evidence_category` | `TEXT` | False | False |
| `measured_quantity` | `TEXT` | False | False |
| `value_json` | `TEXT` | True | False |
| `unit` | `TEXT` | True | False |
| `uncertainty_value` | `REAL` | True | False |
| `measurement_method` | `TEXT` | True | False |
| `test_standard` | `TEXT` | True | False |
| `source_type` | `TEXT` | False | False |
| `source_name` | `TEXT` | True | False |
| `source_record_id` | `TEXT` | True | False |
| `doi_or_url` | `TEXT` | True | False |
| `specimen_id` | `TEXT` | True | False |
| `observed_at` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `validation_status` | `TEXT` | False | False |
| `original_filename` | `TEXT` | True | False |
| `stored_filename` | `TEXT` | True | False |
| `sha256` | `TEXT` | True | False |
| `media_type` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `experiments`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `experiment_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `investigation_id` | `TEXT` | True | False |
| `experiment_type` | `TEXT` | False | False |
| `objective` | `TEXT` | True | False |
| `recommended_measurement` | `TEXT` | True | False |
| `experimental_method` | `TEXT` | True | False |
| `parameters_json` | `TEXT` | False | False |
| `result_json` | `TEXT` | True | False |
| `status` | `TEXT` | False | False |
| `provenance_classification` | `TEXT` | True | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |
| `completed_at` | `TEXT` | True | False |

#### `heat_treatments`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `treatment_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `treatment_type` | `TEXT` | False | False |
| `temperature_value` | `REAL` | True | False |
| `temperature_unit` | `TEXT` | True | False |
| `holding_time` | `REAL` | True | False |
| `holding_time_unit` | `TEXT` | True | False |
| `cooling_method` | `TEXT` | True | False |
| `parameters_json` | `TEXT` | False | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `investigations`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `investigation_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `investigation_goal` | `TEXT` | False | False |
| `status` | `TEXT` | False | False |
| `initial_state_json` | `TEXT` | False | False |
| `final_state_json` | `TEXT` | True | False |
| `limitations_json` | `TEXT` | False | False |
| `started_at` | `TEXT` | False | False |
| `completed_at` | `TEXT` | True | False |

#### `material_compositions`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `composition_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `element` | `TEXT` | False | False |
| `value` | `REAL` | False | False |
| `unit` | `TEXT` | False | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `materials`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `material_id` | `TEXT` | True | True |
| `material_family` | `TEXT` | True | False |
| `grade` | `TEXT` | True | False |
| `designation` | `TEXT` | True | False |
| `description` | `TEXT` | True | False |
| `identity_status` | `TEXT` | False | False |
| `metadata_json` | `TEXT` | False | False |
| `created_at` | `TEXT` | False | False |

#### `microstructure_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `microstructure_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `characterization_method` | `TEXT` | False | False |
| `image_path` | `TEXT` | True | False |
| `map_path` | `TEXT` | True | False |
| `features_json` | `TEXT` | False | False |
| `phases_json` | `TEXT` | False | False |
| `grain_size_value` | `REAL` | True | False |
| `grain_size_unit` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `model_versions`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `model_version_id` | `TEXT` | True | True |
| `model_key` | `TEXT` | False | False |
| `model_name` | `TEXT` | False | False |
| `version_label` | `TEXT` | False | False |
| `artifact_sha256` | `TEXT` | False | False |
| `artifact_path` | `TEXT` | True | False |
| `framework` | `TEXT` | True | False |
| `target_property` | `TEXT` | True | False |
| `metadata_json` | `TEXT` | False | False |
| `registered_at` | `TEXT` | False | False |

#### `prediction_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `prediction_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `investigation_id` | `TEXT` | True | False |
| `target_property` | `TEXT` | False | False |
| `predicted_value` | `REAL` | True | False |
| `predicted_unit` | `TEXT` | True | False |
| `model_name` | `TEXT` | False | False |
| `model_version` | `TEXT` | False | False |
| `input_json` | `TEXT` | False | False |
| `applicability_json` | `TEXT` | False | False |
| `provenance_classification` | `TEXT` | False | False |
| `created_at` | `TEXT` | False | False |
| `prediction_type` | `TEXT` | False | False |
| `prediction_json` | `TEXT` | False | False |

#### `process_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `process_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `process_type` | `TEXT` | False | False |
| `process_parameters_json` | `TEXT` | False | False |
| `sequence_number` | `INTEGER` | True | False |
| `started_at` | `TEXT` | True | False |
| `completed_at` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `property_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `property_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `property_name` | `TEXT` | False | False |
| `value_json` | `TEXT` | False | False |
| `unit` | `TEXT` | True | False |
| `uncertainty_value` | `REAL` | True | False |
| `measurement_method` | `TEXT` | True | False |
| `test_standard` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `schema_migrations`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `version` | `INTEGER` | True | True |
| `description` | `TEXT` | False | False |
| `applied_at` | `TEXT` | False | False |

#### `service_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `service_record_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `specimen_id` | `TEXT` | True | False |
| `environment_json` | `TEXT` | False | False |
| `temperature_value` | `REAL` | True | False |
| `temperature_unit` | `TEXT` | True | False |
| `duration_value` | `REAL` | True | False |
| `duration_unit` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `source_id` | `TEXT` | True | False |
| `evidence_id` | `TEXT` | True | False |
| `created_at` | `TEXT` | False | False |

#### `sources`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `source_id` | `TEXT` | True | True |
| `source_type` | `TEXT` | False | False |
| `source_name` | `TEXT` | False | False |
| `dataset_name` | `TEXT` | True | False |
| `doi_or_url` | `TEXT` | True | False |
| `publication` | `TEXT` | True | False |
| `original_record_id` | `TEXT` | True | False |
| `method` | `TEXT` | True | False |
| `provenance_classification` | `TEXT` | False | False |
| `metadata_json` | `TEXT` | False | False |
| `created_at` | `TEXT` | False | False |

#### `uncertainty_records`

| Column | Type | Nullable | Primary key |
|---|---|---:|---:|
| `uncertainty_id` | `TEXT` | True | True |
| `material_id` | `TEXT` | False | False |
| `investigation_id` | `TEXT` | True | False |
| `prediction_id` | `TEXT` | True | False |
| `uncertainty_category` | `TEXT` | False | False |
| `uncertainty_value` | `REAL` | True | False |
| `uncertainty_unit` | `TEXT` | True | False |
| `uncertainty_kind` | `TEXT` | False | False |
| `lower_bound` | `REAL` | True | False |
| `upper_bound` | `REAL` | True | False |
| `methodology` | `TEXT` | True | False |
| `calibration_status` | `TEXT` | False | False |
| `metadata_json` | `TEXT` | False | False |
| `created_at` | `TEXT` | False | False |

## Data integrity principles

- Preserve original source and record identifiers where available.
- Keep experimental, computational, derived, imputed, and predicted values distinct.
- Do not silently replace missing measurements with imputed values.
- Preserve units and document transformations.
- Keep model version and provenance associated with predictions.
- Treat uploaded evidence as evidence, not automatically as validated scientific truth.

## Migration history

- `app/schemas/__init__.py`
- `app/schemas/aflow.py`
- `app/schemas/api_responses.py`
- `app/schemas/damage_condition.py`
- `app/schemas/evidence.py`
- `app/schemas/investigation.py`
- `app/schemas/jarvis.py`
- `app/schemas/local_dataset.py`
- `app/schemas/material_identity.py`
- `app/schemas/material_state.py`
- `app/schemas/prediction_responses.py`
- `app/schemas/predictions.py`
- `app/schemas/steelbench.py`
- `app/schemas/zenodo.py`
- `app/services/matx_database_service.py`
