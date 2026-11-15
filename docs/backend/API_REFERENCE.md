# MAT-X Backend API Reference

> Generated from the live FastAPI OpenAPI schema on 2026-11-15T07:38:34+05:00.
> This document describes registered routes, not planned or hypothetical routes.

## API overview

- OpenAPI version: `3.1.0`
- API title: `MAT-X API`
- API version: `1.0.0`
- Registered paths: **35**
- Interactive documentation: `/docs`
- OpenAPI JSON: `/openapi.json`

## Endpoint index

| Method | Path | Summary | Tags |
|---|---|---|---|
| `GET` | `/` | Root | — |
| `POST` | `/api/v1/damage/assess` | Damage Condition Assessment | — |
| `GET` | `/api/v1/datasets/aflow/search` | Aflow Search | — |
| `GET` | `/api/v1/datasets/jarvis/structures` | Jarvis Structures | — |
| `GET` | `/api/v1/datasets/local` | Local Dataset List | — |
| `GET` | `/api/v1/datasets/local/records` | Local Dataset Search | — |
| `GET` | `/api/v1/datasets/steelbench/records` | Steelbench Records | — |
| `GET` | `/api/v1/datasets/zenodo/search` | Zenodo Search | — |
| `POST` | `/api/v1/engineering/report` | Engineering Report | — |
| `POST` | `/api/v1/evidence` | Create Evidence | — |
| `GET` | `/api/v1/evidence` | List Evidence | — |
| `POST` | `/api/v1/evidence/upload` | Upload Evidence | — |
| `GET` | `/api/v1/evidence/{evidence_id}` | Retrieve Evidence | — |
| `GET` | `/api/v1/evidence/{evidence_id}/file` | Download Evidence File | — |
| `GET` | `/api/v1/fatigue/response` | Fatigue Response | — |
| `GET` | `/api/v1/health` | Health | — |
| `POST` | `/api/v1/investigation/recommend` | Investigation Recommendation | — |
| `POST` | `/api/v1/investigation/recommend/persist` | Persist Investigation Recommendation | — |
| `POST` | `/api/v1/material/identity/candidates` | Material Identity Candidates | — |
| `POST` | `/api/v1/material/state/reconstruct` | Material State Reconstruction | — |
| `POST` | `/api/v1/microstructure/analyze` | Analyze Microstructure Upload | — |
| `GET` | `/api/v1/microstructure/maps` | Microstructure Maps | — |
| `GET` | `/api/v1/microstructure/maps/{map_id}/features` | Microstructure Features | — |
| `GET` | `/api/v1/microstructure/maps/{map_id}/unified` | Unified Microstructure Record | — |
| `GET` | `/api/v1/microstructure/provenance` | Microstructure Provenance | — |
| `GET` | `/api/v1/models` | Models | — |
| `POST` | `/api/v1/predict/inclusion` | Inclusion | — |
| `POST` | `/api/v1/predict/tensile-strength` | Tensile Strength | — |
| `POST` | `/api/v1/predict/yield-strength` | Yield Strength | — |
| `GET` | `/api/v1/process/history` | Process History | — |
| `GET` | `/api/v1/process/history/analysis` | Process History Analysis | — |
| `GET` | `/api/v1/process/history/grade/{grade_id}` | Process History By Grade | — |
| `GET` | `/api/v1/process/history/heat/{heat_id}` | Process History By Heat | — |
| `POST` | `/api/v1/scientific-records/{table}` | Create Scientific Record Api | — |
| `GET` | `/api/v1/scientific-records/{table}` | List Scientific Records Api | — |
| `GET` | `/api/v1/scientific-records/{table}/{record_id}` | Retrieve Scientific Record Api | — |
| `PATCH` | `/api/v1/scientific-records/{table}/{record_id}/lifecycle` | Update Scientific Lifecycle Api | — |

## Endpoint details

### GET `/`

Root

- Operation ID: `root__get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: Object / schema not specified |

---

### POST `/api/v1/damage/assess`

Damage Condition Assessment

- Operation ID: `damage_condition_assessment_api_v1_damage_assess_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `DamageAssessmentInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: DamageAssessmentResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/datasets/aflow/search`

Aflow Search

- Operation ID: `aflow_search_api_v1_datasets_aflow_search_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `elements` | `query` | True | array of string | Chemical element symbols, e.g. Fe and Ni. |
| `page` | `query` | False | integer | — |
| `page_size` | `query` | False | integer | — |
| `exact_species_count` | `query` | False | integer or null | — |
| `require_properties` | `query` | False | boolean | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: AflowSearchResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/datasets/jarvis/structures`

Jarvis Structures

- Operation ID: `jarvis_structures_api_v1_datasets_jarvis_structures_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `elements` | `query` | True | array of string | Chemical element symbols, e.g. Fe and Ni. |
| `page_limit` | `query` | False | integer | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: JarvisSearchResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/datasets/local`

Local Dataset List

- Operation ID: `local_dataset_list_api_v1_datasets_local_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: LocalDatasetListResponse |

---

### GET `/api/v1/datasets/local/records`

Local Dataset Search

- Operation ID: `local_dataset_search_api_v1_datasets_local_records_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `filename` | `query` | True | string | — |
| `query` | `query` | False | string or null | — |
| `limit` | `query` | False | integer | — |
| `offset` | `query` | False | integer | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: LocalDatasetSearchResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/datasets/steelbench/records`

Steelbench Records

- Operation ID: `steelbench_records_api_v1_datasets_steelbench_records_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `heat_id` | `query` | False | string or null | — |
| `grade_id` | `query` | False | string or null | — |
| `steel_family` | `query` | False | string or null | — |
| `source` | `query` | False | string or null | — |
| `data_tier` | `query` | False | string or null | — |
| `query` | `query` | False | string or null | — |
| `limit` | `query` | False | integer | — |
| `offset` | `query` | False | integer | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: SteelBenchSearchResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/datasets/zenodo/search`

Zenodo Search

- Operation ID: `zenodo_search_api_v1_datasets_zenodo_search_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `query` | `query` | True | string | — |
| `page` | `query` | False | integer | — |
| `size` | `query` | False | integer | — |
| `sort` | `query` | False | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ZenodoSearchResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/engineering/report`

Engineering Report

- Operation ID: `engineering_report_api_v1_engineering_report_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `MaterialStateReconstructionInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: EngineeringReportResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/evidence`

Create an immutable scientific evidence record.

- Operation ID: `create_evidence_api_v1_evidence_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `EvidenceCreateInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `201` | Successful Response | application/json: EvidenceRecordResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/evidence`

List evidence records with optional filters.

- Operation ID: `list_evidence_api_v1_evidence_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `material_id` | `query` | False | string or null | — |
| `investigation_id` | `query` | False | string or null | — |
| `limit` | `query` | False | integer | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: array of EvidenceRecordResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/evidence/upload`

Upload a scientific evidence file and register its provenance.

- Operation ID: `upload_evidence_api_v1_evidence_upload_post`
- Tags: none

**Request body**

- Content type: `multipart/form-data`
- Schema: `Body_upload_evidence_api_v1_evidence_upload_post`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `201` | Successful Response | application/json: EvidenceRecordResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/evidence/{evidence_id}`

Retrieve one evidence record by its identifier.

- Operation ID: `retrieve_evidence_api_v1_evidence__evidence_id__get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `evidence_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: EvidenceRecordResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/evidence/{evidence_id}/file`

Download an evidence file after path and checksum verification.

- Operation ID: `download_evidence_file_api_v1_evidence__evidence_id__file_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `evidence_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Verified evidence file download. | application/octet-stream: string |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/fatigue/response`

Fatigue Response

- Operation ID: `fatigue_response_api_v1_fatigue_response_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: FatigueResponse |

---

### GET `/api/v1/health`

Health

- Operation ID: `health_api_v1_health_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: HealthResponse |

---

### POST `/api/v1/investigation/recommend`

Investigation Recommendation

- Operation ID: `investigation_recommendation_api_v1_investigation_recommend_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `ActiveInvestigationInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ActiveInvestigationResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/investigation/recommend/persist`

Recommend and persist proposed experiments without claiming execution.

- Operation ID: `persist_investigation_recommendation_api_v1_investigation_recommend_persist_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `ActiveInvestigationInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ActiveInvestigationResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/material/identity/candidates`

Material Identity Candidates

- Operation ID: `material_identity_candidates_api_v1_material_identity_candidates_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `MaterialIdentityInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: MaterialIdentityResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/material/state/reconstruct`

Material State Reconstruction

- Operation ID: `material_state_reconstruction_api_v1_material_state_reconstruct_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `MaterialStateReconstructionInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: MaterialStateResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/microstructure/analyze`

Analyze Microstructure Upload

- Operation ID: `analyze_microstructure_upload_api_v1_microstructure_analyze_post`
- Tags: none

**Request body**

- Content type: `multipart/form-data`
- Schema: `Body_analyze_microstructure_upload_api_v1_microstructure_analyze_post`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: Object / schema not specified |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/microstructure/maps`

Microstructure Maps

- Operation ID: `microstructure_maps_api_v1_microstructure_maps_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: MicrostructureMapsResponse |

---

### GET `/api/v1/microstructure/maps/{map_id}/features`

Microstructure Features

- Operation ID: `microstructure_features_api_v1_microstructure_maps__map_id__features_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `map_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: object |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/microstructure/maps/{map_id}/unified`

Unified Microstructure Record

- Operation ID: `unified_microstructure_record_api_v1_microstructure_maps__map_id__unified_get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `map_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: UnifiedMicrostructureResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/microstructure/provenance`

Microstructure Provenance

- Operation ID: `microstructure_provenance_api_v1_microstructure_provenance_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: EbsdProvenanceResponse |

---

### GET `/api/v1/models`

Models

- Operation ID: `models_api_v1_models_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ModelsResponse |

---

### POST `/api/v1/predict/inclusion`

Inclusion

- Operation ID: `inclusion_api_v1_predict_inclusion_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `InclusionPredictionInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: InclusionPredictionResponse |
| `503` | Prediction model or persistence service unavailable | No response body schema declared |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/predict/tensile-strength`

Tensile Strength

- Operation ID: `tensile_strength_api_v1_predict_tensile_strength_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `TensilePredictionInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: TensilePredictionResponse |
| `503` | Prediction model or persistence service unavailable | No response body schema declared |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/predict/yield-strength`

Yield Strength

- Operation ID: `yield_strength_api_v1_predict_yield_strength_post`
- Tags: none

**Request body**

- Content type: `application/json`
- Schema: `YieldPredictionInput`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: YieldPredictionResponse |
| `503` | Prediction model or persistence service unavailable | No response body schema declared |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/process/history`

Process History

- Operation ID: `process_history_api_v1_process_history_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ProcessHistoryResponse |

---

### GET `/api/v1/process/history/analysis`

Process History Analysis

- Operation ID: `process_history_analysis_api_v1_process_history_analysis_get`
- Tags: none

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ProcessHistoryAnalysisResponse |

---

### GET `/api/v1/process/history/grade/{grade_id}`

Process History By Grade

- Operation ID: `process_history_by_grade_api_v1_process_history_grade__grade_id__get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `grade_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ProcessLookupResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/process/history/heat/{heat_id}`

Process History By Heat

- Operation ID: `process_history_by_heat_api_v1_process_history_heat__heat_id__get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `heat_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: ProcessLookupResponse |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### POST `/api/v1/scientific-records/{table}`

Create an append-only record in an allowed scientific table.

- Operation ID: `create_scientific_record_api_api_v1_scientific_records__table__post`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `table` | `path` | True | string | — |

**Request body**

- Content type: `application/json`
- Schema: `object`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `201` | Successful Response | application/json: object |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/scientific-records/{table}`

List scientific records using optional JSON equality filters.

- Operation ID: `list_scientific_records_api_api_v1_scientific_records__table__get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `table` | `path` | True | string | — |
| `filters_json` | `query` | False | string or null | — |
| `limit` | `query` | False | integer | — |
| `offset` | `query` | False | integer | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: array of object |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### GET `/api/v1/scientific-records/{table}/{record_id}`

Retrieve a scientific record by its table-specific identifier.

- Operation ID: `retrieve_scientific_record_api_api_v1_scientific_records__table___record_id__get`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `table` | `path` | True | string | — |
| `record_id` | `path` | True | string | — |

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: object |
| `422` | Validation Error | application/json: HTTPValidationError |

---

### PATCH `/api/v1/scientific-records/{table}/{record_id}/lifecycle`

Advance an investigation or experiment through valid lifecycle states.

- Operation ID: `update_scientific_lifecycle_api_api_v1_scientific_records__table___record_id__lifecycle_patch`
- Tags: none

**Parameters**

| Name | Location | Required | Type | Description |
|---|---|---:|---|---|
| `table` | `path` | True | string | — |
| `record_id` | `path` | True | string | — |

**Request body**

- Content type: `application/json`
- Schema: `object`

**Responses**

| Status | Description | Response schema |
|---|---|---|
| `200` | Successful Response | application/json: object |
| `422` | Validation Error | application/json: HTTPValidationError |

---

## Schema components

The authoritative request and response models are included in `openapi.json`.
Inspect `components.schemas` for full field definitions, constraints, enums, and examples.

## Integration rule

Frontend developers must use the paths, HTTP methods, request models, and response models listed here and in `openapi.json`. Do not assume conceptual endpoint names exist unless they appear in this generated reference.
