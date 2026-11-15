# MAT-X Backend Architecture
## 1. Architectural objective
The backend supports reconstruction of an engineering material's current state from
heterogeneous and potentially incomplete evidence. It is designed around an iterative
human-in-the-loop workflow rather than a single property-prediction endpoint.
Conceptual material chain:
`Composition → Processing → Microstructure → Properties → Service → Damage → Material State`
The system uses evidence and model outputs to identify gaps and support further
characterization.
## 2. Main backend layers
### API layer
FastAPI routes are registered through the application router and exposed under `/api/v1`.
The generated `openapi.json` and `API_REFERENCE.md` describe the actual routes.
### Schema layer
Request and response schemas define the API contract, validation constraints, and
structured scientific data exchanged between the frontend and services.
### Service layer
Services contain domain operations, including material-state reconstruction, prediction
persistence, microstructure analysis, evidence handling, process-history retrieval,
investigation recommendations, and engineering-report generation.
### Repository and persistence layer
The repository service provides controlled access to scientific records. Database
initialization and schema migration logic maintain the relational data model. Model
version metadata is persisted separately from individual prediction values.
### Evidence storage
Evidence upload and registry functions maintain evidence metadata and file references.
The evidence registry is initialized through application lifecycle handling.
### Dataset adapters
Source adapters provide a common application-facing interface for local data and external
materials data providers. External services may have availability, rate-limit, or network
constraints.
## 3. Six conceptual MAT-X engines
### Engine 1 — Material Identity
Accepts composition and identity-related evidence and generates candidate material
identity information. Candidate results must not be interpreted as a guaranteed grade
identification.
### Engine 2 — Process History
Represents and retrieves manufacturing and heat-treatment history, including heat- and
grade-related records where supported by the stored data.
### Engine 3 — Microstructure
Accepts micrograph evidence and exposes image analysis, map, feature, and provenance
operations. Current image-analysis outputs must be interpreted within the limitations
documented in `SCIENTIFIC_METHODS_AND_LIMITATIONS.md`.
### Engine 4 — Property Reconstruction
Exposes supported tensile-strength, yield-strength, and inclusion prediction operations.
Prediction records are persisted with model-version metadata where applicable.
### Engine 5 — Damage / Condition
Provides damage assessment and evidence-related operations. The current implementation
must not be interpreted as a validated failure diagnosis or remaining-life system.
### Engine 6 — Active Investigation
Uses available evidence and uncertainty-related information to recommend further
investigation. Recommendation outputs must be interpreted according to their stated
method and validation status.
## 4. Typical request lifecycle
1. The frontend submits a validated request to a versioned API route.
2. FastAPI validates path, query, form, and body inputs.
3. The route delegates work to the relevant service.
4. Services retrieve evidence, scientific records, dataset information, or model outputs.
5. Repository functions persist records where the operation requires persistence.
6. The API returns structured JSON or an explicit HTTP error.
7. The frontend presents the result while preserving provenance and scientific limitations.
## 5. Core design principles
- Separate API transport logic from scientific service logic.
- Keep database access behind repository/service functions.
- Preserve source provenance and original identifiers.
- Distinguish measured values from predictions and derived values.
- Expose unavailable or insufficient evidence instead of inventing values.
- Keep model version information attached to predictions where available.
- Do not treat an LLM as the core numerical scientific model.
- Keep engineering interpretation under human review.
## 6. Runtime and lifecycle
The FastAPI application is imported as `app.main:app`. Application lifecycle handling
initializes the evidence registry and MAT-X database. The API router is mounted with
the `/api/v1` prefix.
For the exact runtime route inventory, use the generated `API_REFERENCE.md`.
