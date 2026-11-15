# MAT-X Version 1.0 ? Backend API Traceability

Source: Generated FastAPI OpenAPI schema from app.main:app

OpenAPI paths discovered: 35

> A functional equivalent is not an identical route. Scientific
> limitations must be assessed separately from API route coverage.

## Requirement mapping

### Material identity
- Coverage: **FUNCTIONAL EQUIVALENT**
- Endpoints:
  - `/api/v1/material/identity/candidates`
- Notes: Candidate-based material identity; not a generic material CRUD endpoint.

### Material state reconstruction
- Coverage: **FUNCTIONAL EQUIVALENT**
- Endpoints:
  - `/api/v1/material/state/reconstruct`
- Notes: Reconstruction endpoint; no generic /materials/{id}/state route.

### Material listing, creation and individual retrieval
- Coverage: **FUNCTIONAL EQUIVALENT ? IMPLEMENTED**
- Endpoints:
  - `POST /api/v1/scientific-records/{table}` with `table=materials` ? create material.
  - `GET /api/v1/scientific-records/{table}` with `table=materials` ? list and filter materials.
  - `GET /api/v1/scientific-records/{table}/{record_id}` with `table=materials` ? retrieve material.
- Notes: Generic scientific-record endpoints provide material CRUD operations. Dedicated `/materials` routes are not required for this implementation.
- Notes: Do not infer generic material CRUD from reconstruction or evidence routes.

### Prediction services
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/predict/tensile-strength`
  - `/api/v1/predict/yield-strength`
  - `/api/v1/predict/inclusion`
- Notes: Separate tensile, yield and inclusion prediction endpoints.

### Microstructure analysis
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/microstructure/analyze`
  - `/api/v1/microstructure/maps`
  - `/api/v1/microstructure/provenance`
- Notes: Image analysis and EBSD map-related endpoints.

### Uncertainty-aware prediction
- Coverage: **FUNCTIONAL SUPPORT; NO STANDALONE UNCERTAINTY ROUTE**
- Endpoints:
  - `/api/v1/predict/tensile-strength`
  - `/api/v1/predict/yield-strength`
  - `/api/v1/predict/inclusion`
- Notes: Prediction contracts must be checked for uncertainty fields and limitations.

### Active investigation
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/investigation/recommend`
  - `/api/v1/investigation/recommend/persist`
- Notes: Recommendation and persistence endpoints; physical validation remains limited.

### Next-best measurement
- Coverage: **FUNCTIONAL EQUIVALENT**
- Endpoints:
  - `/api/v1/investigation/recommend`
- Notes: Investigation recommendation is the implemented workflow; no separate next-measurement route.

### Dataset integrations
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/datasets/local`
  - `/api/v1/datasets/local/records`
  - `/api/v1/datasets/steelbench/records`
  - `/api/v1/datasets/zenodo/search`
  - `/api/v1/datasets/jarvis/structures`
  - `/api/v1/datasets/aflow/search`
- Notes: Local, SteelBench, Zenodo, JARVIS and AFLOW adapters.

### Evidence and provenance
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/evidence`
  - `/api/v1/evidence/upload`
  - `/api/v1/evidence/{evidence_id}`
  - `/api/v1/evidence/{evidence_id}/file`
  - `/api/v1/microstructure/provenance`
- Notes: Evidence registry, file access and microstructure provenance.

### Model registry
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/models`
- Notes: Model metadata endpoint.

### Health
- Coverage: **DIRECT**
- Endpoints:
  - `/api/v1/health`
- Notes: API health endpoint.

## Complete registered OpenAPI inventory

- `GET /` ? Root
- `POST /api/v1/damage/assess` ? Damage Condition Assessment
- `GET /api/v1/datasets/aflow/search` ? Aflow Search
- `GET /api/v1/datasets/jarvis/structures` ? Jarvis Structures
- `GET /api/v1/datasets/local` ? Local Dataset List
- `GET /api/v1/datasets/local/records` ? Local Dataset Search
- `GET /api/v1/datasets/steelbench/records` ? Steelbench Records
- `GET /api/v1/datasets/zenodo/search` ? Zenodo Search
- `POST /api/v1/engineering/report` ? Engineering Report
- `GET /api/v1/evidence` ? List Evidence
- `POST /api/v1/evidence` ? Create Evidence
- `POST /api/v1/evidence/upload` ? Upload Evidence
- `GET /api/v1/evidence/{evidence_id}` ? Retrieve Evidence
- `GET /api/v1/evidence/{evidence_id}/file` ? Download Evidence File
- `GET /api/v1/fatigue/response` ? Fatigue Response
- `GET /api/v1/health` ? Health
- `POST /api/v1/investigation/recommend` ? Investigation Recommendation
- `POST /api/v1/investigation/recommend/persist` ? Persist Investigation Recommendation
- `POST /api/v1/material/identity/candidates` ? Material Identity Candidates
- `POST /api/v1/material/state/reconstruct` ? Material State Reconstruction
- `POST /api/v1/microstructure/analyze` ? Analyze Microstructure Upload
- `GET /api/v1/microstructure/maps` ? Microstructure Maps
- `GET /api/v1/microstructure/maps/{map_id}/features` ? Microstructure Features
- `GET /api/v1/microstructure/maps/{map_id}/unified` ? Unified Microstructure Record
- `GET /api/v1/microstructure/provenance` ? Microstructure Provenance
- `GET /api/v1/models` ? Models
- `POST /api/v1/predict/inclusion` ? Inclusion
- `POST /api/v1/predict/tensile-strength` ? Tensile Strength
- `POST /api/v1/predict/yield-strength` ? Yield Strength
- `GET /api/v1/process/history` ? Process History
- `GET /api/v1/process/history/analysis` ? Process History Analysis
- `GET /api/v1/process/history/grade/{grade_id}` ? Process History By Grade
- `GET /api/v1/process/history/heat/{heat_id}` ? Process History By Heat
- `GET /api/v1/scientific-records/{table}` ? List Scientific Records Api
- `POST /api/v1/scientific-records/{table}` ? Create Scientific Record Api
- `GET /api/v1/scientific-records/{table}/{record_id}` ? Retrieve Scientific Record Api
- `PATCH /api/v1/scientific-records/{table}/{record_id}/lifecycle` ? Update Scientific Lifecycle Api

## Material API contract clarification
The Version 1.0 logical material operations are implemented through the
generic scientific-record repository API. The `materials` table is the
resource selector. This avoids duplicating material persistence routes.
Verified operations:
- Create: `POST /api/v1/scientific-records/materials`
- List/filter: `GET /api/v1/scientific-records/materials`
- Retrieve: `GET /api/v1/scientific-records/materials/{record_id}`
- Reconstruct: `POST /api/v1/material/state/reconstruct`
