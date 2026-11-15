from pathlib import Path
import json
import os
import re
import sys
import sqlite3
import datetime
import importlib
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "backend"
DOCS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
TODAY = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
def write_doc(name, content):
    path = DOCS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"CREATED: {path.relative_to(ROOT)}")
def relative_files(directory, extensions=None):
    base = ROOT / directory
    if not base.exists():
        return []
    ignored = {".venv", "venv", "__pycache__", "node_modules", ".git"}
    result = []
    for p in base.rglob("*"):
        if any(part in ignored for part in p.parts):
            continue
        if p.is_file() and (extensions is None or p.suffix.lower() in extensions):
            result.append(p.relative_to(ROOT).as_posix())
    return sorted(result)
def resolve_ref(schema):
    if not isinstance(schema, dict):
        return "Not specified"
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    if "type" in schema:
        value = schema["type"]
        if value == "array":
            return "array of " + resolve_ref(schema.get("items", {}))
        return value
    if "anyOf" in schema:
        return " or ".join(resolve_ref(x) for x in schema["anyOf"])
    if "oneOf" in schema:
        return " or ".join(resolve_ref(x) for x in schema["oneOf"])
    return "Object / schema not specified"
def generate_openapi():
    try:
        from app.main import app
        schema = app.openapi()
        (DOCS / "openapi.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print("OPENAPI: generated from app.main:app")
        return schema
    except Exception as exc:
        message = (
            "# OpenAPI generation error\n\n"
            "The application could not generate its OpenAPI schema in this environment.\n\n"
            f"Error: `{type(exc).__name__}: {exc}`\n\n"
            "Resolve the import/configuration issue, then rerun "
            "`python scripts/generate_backend_docs.py`.\n"
        )
        write_doc("OPENAPI_GENERATION_ERROR.md", message)
        print(f"OPENAPI WARNING: {type(exc).__name__}: {exc}")
        return None
def generate_api_reference(schema):
    if not schema:
        write_doc("API_REFERENCE.md", """
# MAT-X Backend API Reference
OpenAPI generation did not complete. See `OPENAPI_GENERATION_ERROR.md`.
The API reference must be regenerated after the application import issue is resolved.
""")
        return
    paths = schema.get("paths", {})
    lines = [
        "# MAT-X Backend API Reference",
        "",
        f"> Generated from the live FastAPI OpenAPI schema on {TODAY}.",
        "> This document describes registered routes, not planned or hypothetical routes.",
        "",
        "## API overview",
        "",
        f"- OpenAPI version: `{schema.get('openapi', 'unknown')}`",
        f"- API title: `{schema.get('info', {}).get('title', 'MAT-X API')}`",
        f"- API version: `{schema.get('info', {}).get('version', 'unspecified')}`",
        f"- Registered paths: **{len(paths)}**",
        "- Interactive documentation: `/docs`",
        "- OpenAPI JSON: `/openapi.json`",
        "",
        "## Endpoint index",
        "",
        "| Method | Path | Summary | Tags |",
        "|---|---|---|---|",
    ]
    operations = []
    for path, path_item in sorted(paths.items()):
        for method, operation in path_item.items():
            if method.lower() not in {
                "get", "post", "put", "patch", "delete", "options", "head"
            }:
                continue
            summary = operation.get("summary") or operation.get("operationId") or "—"
            tags = ", ".join(operation.get("tags", [])) or "—"
            lines.append(
                f"| `{method.upper()}` | `{path}` | "
                f"{summary} | {tags} |"
            )
            operations.append((path, method.upper(), operation))
    lines.extend(["", "## Endpoint details", ""])
    for path, method, operation in operations:
        lines.extend([
            f"### {method} `{path}`",
            "",
            operation.get("description", operation.get("summary", "No description supplied.")),
            "",
            f"- Operation ID: `{operation.get('operationId', 'not supplied')}`",
            f"- Tags: {', '.join(operation.get('tags', [])) or 'none'}",
            "",
        ])
        params = operation.get("parameters", [])
        if params:
            lines.extend([
                "**Parameters**",
                "",
                "| Name | Location | Required | Type | Description |",
                "|---|---|---:|---|---|",
            ])
            for p in params:
                lines.append(
                    f"| `{p.get('name', '')}` | `{p.get('in', '')}` | "
                    f"{p.get('required', False)} | "
                    f"{resolve_ref(p.get('schema', {}))} | "
                    f"{str(p.get('description', '—'))} |"
                )
            lines.append("")
        request_body = operation.get("requestBody", {})
        if request_body:
            lines.extend(["**Request body**", ""])
            for content_type, body in request_body.get("content", {}).items():
                lines.append(
                    f"- Content type: `{content_type}`"
                )
                lines.append(
                    f"- Schema: `{resolve_ref(body.get('schema', {}))}`"
                )
            lines.append("")
        responses = operation.get("responses", {})
        if responses:
            lines.extend([
                "**Responses**",
                "",
                "| Status | Description | Response schema |",
                "|---|---|---|",
            ])
            for status, response in responses.items():
                response_content = response.get("content", {})
                schemas = []
                for media_type, body in response_content.items():
                    schemas.append(
                        f"{media_type}: {resolve_ref(body.get('schema', {}))}"
                    )
                lines.append(
                    f"| `{status}` | "
                    f"{str(response.get('description', '—'))} | "
                    f"{'; '.join(schemas) if schemas else 'No response body schema declared'} |"
                )
            lines.append("")
        lines.append("---")
        lines.append("")
    lines.extend([
        "## Schema components",
        "",
        "The authoritative request and response models are included in `openapi.json`.",
        "Inspect `components.schemas` for full field definitions, constraints, enums, and examples.",
        "",
        "## Integration rule",
        "",
        "Frontend developers must use the paths, HTTP methods, request models, and response "
        "models listed here and in `openapi.json`. Do not assume conceptual endpoint names "
        "exist unless they appear in this generated reference.",
        "",
    ])
    write_doc("API_REFERENCE.md", "\n".join(lines))
def generate_database_inventory():
    lines = [
        "# MAT-X Database and Data Model",
        "",
        f"> Generated on {TODAY}. Database inspection reads schema metadata only; "
        "it does not export application records.",
        "",
        "## Logical entities from the Version 1.0 design",
        "",
        "| Entity | Purpose |",
        "|---|---|",
        "| materials | Material identity and investigation-level records |",
        "| material_compositions | Chemical composition evidence |",
        "| process_records | Manufacturing and process history |",
        "| heat_treatments | Heat-treatment conditions and history |",
        "| microstructure_records | Microstructure analysis and associated metadata |",
        "| property_records | Observed or recorded material properties |",
        "| service_records | Service and operating conditions |",
        "| damage_records | Damage-related observations |",
        "| prediction_records | Model-generated predictions |",
        "| uncertainty_records | Uncertainty and applicability information |",
        "| investigations | Investigation and recommendation records |",
        "| sources | Dataset and source metadata |",
        "| experiments | Experiment or evaluation records |",
        "| model_versions | Registered model version metadata |",
        "| evidence registry | Uploaded evidence metadata and file references |",
        "",
        "The logical design in the specification and the implemented physical schema "
        "are not assumed to be identical. The following section reports tables actually "
        "found in local SQLite databases.",
        "",
        "## Implemented SQLite schema inventory",
        "",
    ]
    candidates = set()
    for folder in ["data", "storage", ""]:
        base = ROOT / folder
        if base.exists():
            for p in base.rglob("*.db"):
                if any(x in {".venv", "venv", "node_modules", ".git", "__pycache__"}
                       for x in p.parts):
                    continue
                candidates.add(p.resolve())
    try:
        paths_module = importlib.import_module("app.core.paths")
        for key in ["EVIDENCE_DB_PATH", "DATABASE_PATH", "MATX_DB_PATH", "DB_PATH"]:
            value = getattr(paths_module, key, None)
            if value:
                p = Path(value)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    candidates.add(p.resolve())
    except Exception:
        pass
    if not candidates:
        lines.append(
            "No local `.db` files were found in the inspected data/storage locations. "
            "The database may be created during application startup or configured externally."
        )
    for db_path in sorted(candidates):
        lines.extend([
            f"### `{db_path.relative_to(ROOT).as_posix() if db_path.is_relative_to(ROOT) else db_path.as_posix()}`",
            "",
        ])
        try:
            uri = db_path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            tables = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            if not tables:
                lines.append("No application tables found.")
            for (table,) in tables:
                lines.extend([f"#### `{table}`", "", "| Column | Type | Nullable | Primary key |", "|---|---|---:|---:|"])
                for col in connection.execute(f'PRAGMA table_info("{table}")').fetchall():
                    _, name, col_type, notnull, default, pk = col
                    lines.append(
                        f"| `{name}` | `{col_type or 'unspecified'}` | "
                        f"{not bool(notnull)} | {bool(pk)} |"
                    )
                lines.append("")
            connection.close()
        except Exception as exc:
            lines.append(f"Schema inspection unavailable: `{type(exc).__name__}: {exc}`")
            lines.append("")
    lines.extend([
        "## Data integrity principles",
        "",
        "- Preserve original source and record identifiers where available.",
        "- Keep experimental, computational, derived, imputed, and predicted values distinct.",
        "- Do not silently replace missing measurements with imputed values.",
        "- Preserve units and document transformations.",
        "- Keep model version and provenance associated with predictions.",
        "- Treat uploaded evidence as evidence, not automatically as validated scientific truth.",
        "",
        "## Migration history",
        "",
    ])
    migration_files = relative_files("app", {".py", ".sql"})
    migration_files = [
        x for x in migration_files
        if "migration" in x.lower() or "database" in x.lower() or "schema" in x.lower()
    ]
    if migration_files:
        lines.extend(f"- `{x}`" for x in migration_files)
    else:
        lines.append("Migration source files were not automatically identified; consult the database service.")
    write_doc("DATA_MODEL.md", "\n".join(lines))
def main():
    print("\nMAT-X BACKEND DOCUMENTATION GENERATOR")
    print("=" * 55)
    write_doc("README.md", f"""
# MAT-X Backend Documentation
**Project:** MAT-X — AI-Assisted Materials State Reconstruction and Active Investigation Engine  
**Specification baseline:** Project Specification, Version 1.0  
**Documentation generated:** {TODAY}
## Purpose
MAT-X is a human-in-the-loop engineering investigation platform. Its backend combines
material identity, process history, microstructure evidence, property records, condition
evidence, uncertainty/applicability information, and investigation recommendations into
a structured material-state workflow.
The primary Version 1.0 domain is engineering steels. The architecture is intended to
remain extensible to other material families.
## Documentation index
| Document | Contents |
|---|---|
| [Architecture](ARCHITECTURE.md) | Backend components, engines, and request lifecycle |
| [Getting Started](GETTING_STARTED.md) | Local setup, startup, and development workflow |
| [API Reference](API_REFERENCE.md) | Actual registered routes and OpenAPI-derived details |
| [OpenAPI JSON](openapi.json) | Machine-readable API contract for frontend integration |
| [Database and Data Model](DATA_MODEL.md) | Logical entities and discovered physical SQLite schema |
| [Data Sources and Provenance](DATA_SOURCES_AND_PROVENANCE.md) | Dataset adapters, value categories, and traceability |
| [Scientific Methods and Limitations](SCIENTIFIC_METHODS_AND_LIMITATIONS.md) | What the current backend does and does not claim |
| [Configuration and Operations](CONFIGURATION_AND_OPERATIONS.md) | Environment, run commands, logs, and operational guidance |
| [Frontend Handoff](FRONTEND_HANDOFF.md) | API integration rules and examples |
| [Specification Traceability](SPECIFICATION_TRACEABILITY.md) | Version 1.0 requirements mapped to implementation evidence |
| [Testing](TESTING.md) | Regression test instructions and acceptance evidence |
## Technology
The backend is implemented in Python using FastAPI. The application exposes a versioned
API under `/api/v1`. Database initialization and evidence-registry initialization are
performed by the application lifecycle.
The generated OpenAPI file is the authoritative API contract for frontend development.
## Quick start
From the repository root, activate the project's virtual environment and install the
dependencies declared by the repository. Then run:
```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.
## Quality baseline
The latest user-provided regression output before this documentation generation reported
**463 passed tests and 5 warnings**. Rerun the suite using the Testing instructions to
confirm the current checkout.
## Important scientific boundary
MAT-X is an AI-assisted engineering investigation system. It is not a certified
engineering decision system, guaranteed failure-diagnosis tool, remaining-useful-life
predictor, or replacement for laboratory testing or qualified engineers.
## Documentation maintenance
After changing routes or schemas, regenerate the API documentation:
```powershell
.\\.venv\\Scripts\\python.exe scripts\\generate_backend_docs.py
```
Review the generated files before committing or sharing them.
""")
    write_doc("ARCHITECTURE.md", """
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
""")
    write_doc("GETTING_STARTED.md", """
# Getting Started — MAT-X Backend
## 1. Requirements
- Windows, Linux, or macOS development environment.
- Python version compatible with the repository dependencies.
- Project virtual environment.
- Dependencies declared in `requirements.txt` or `pyproject.toml`, depending on the
  repository's package configuration.
- Network access for any external dataset adapter that requires it.
## 2. Open the project
```powershell
cd C:\\Users\\Ahmadcomputer\\MAT-X
```
## 3. Activate the virtual environment
```powershell
.\\.venv\\Scripts\\Activate.ps1
```
If PowerShell blocks activation, use the interpreter directly:
```powershell
.\\.venv\\Scripts\\python.exe --version
```
## 4. Install dependencies
Use the dependency file that exists in the repository:
```powershell
if (Test-Path requirements.txt) {
    .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
}
elseif (Test-Path pyproject.toml) {
    .\\.venv\\Scripts\\python.exe -m pip install -e .
}
else {
    Write-Error "No requirements.txt or pyproject.toml found."
}
```
## 5. Configure environment
Review `.env.example`, `.env.template`, or the project's configuration module if present.
Create a local `.env` file only with the variables required by the current application.
Never commit secrets, API keys, credentials, or private database URLs.
Do not copy production credentials into frontend code.
## 6. Start the API
```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
```
Default local address:
`http://127.0.0.1:8000`
## 7. Verify the service
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health endpoint: consult the generated API reference for the exact path and response.
## 8. Run tests
```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```
## 9. Regenerate documentation
```powershell
.\\.venv\\Scripts\\python.exe scripts\\generate_backend_docs.py
```
Run the generator from the project root so that imports and relative paths resolve.
""")
    write_doc("DATA_SOURCES_AND_PROVENANCE.md", """
# Data Sources, Adapters, and Provenance
## 1. Data-source strategy
MAT-X Version 1.0 identifies public materials data as its primary data strategy. The
backend includes adapter endpoints for the following source categories:
| Source | Intended role | Data interpretation |
|---|---|---|
| SteelBench | Experimental steel-property and heat-level data | Experimental records, subject to source metadata |
| NIST JARVIS | Computational materials knowledge | Computational values must remain distinguishable |
| AFLOW | Computational materials information | Computational values must remain distinguishable |
| Zenodo | Public research datasets | Dataset-specific metadata and licensing apply |
| Local datasets | User-provided CSV/XLSX data | Requires validation and source attribution |
The existence of an adapter does not mean every record from the source has been
downloaded, ingested, validated, or persisted locally.
## 2. Provenance fields
The Version 1.0 specification calls for preservation of:
- Original source.
- Dataset name.
- DOI or source URL.
- Original record identifier.
- Material identifier.
- Measurement method.
- Publication or source reference.
- Units.
- Transformation history.
- Experimental or computational classification.
- Original or imputed classification.
Where a field is not available from a source, it should be reported as unavailable
rather than fabricated.
## 3. Scientific value categories
MAT-X must keep these categories distinct:
| Category | Meaning |
|---|---|
| Experimental | Measured experimentally |
| Computational | Obtained through DFT, simulation, or another computational method |
| Derived | Calculated from other values |
| Imputed | Estimated to replace a missing value |
| Predicted | Produced by a MAT-X model |
A prediction must never be presented as an experimental measurement.
## 4. Local data lake
The standardized project data layout includes:
```text
data/
├── raw/
│   ├── steelbench/
│   ├── microsteel/
│   ├── 316l/
│   └── inclusions/
├── processed/
│   ├── normalized/
│   ├── deduplicated/
│   └── validated/
├── features/
│   ├── tabular/
│   └── image/
├── models/
└── metadata/
```
The raw layer is intended to preserve original downloaded files. Processed and feature
layers should retain traceability to their source records and transformation steps.
The presence of a directory does not imply that the corresponding dataset has been
fully downloaded or processed.
## 5. External source reliability
External API responses may be unavailable, incomplete, rate-limited, or changed by
their providers. Clients should handle source errors and incomplete results explicitly.
Do not interpret an empty response as proof that a material or property does not exist.
## 6. Dataset licensing
Before redistributing a dataset or its derived files, verify its license and citation
requirements at the original provider. This documentation does not grant redistribution
rights.
""")
    write_doc("SCIENTIFIC_METHODS_AND_LIMITATIONS.md", """
# Scientific Methods, Interpretation, and Limitations
## 1. Purpose
This document defines the boundary between what the MAT-X backend exposes and what
an engineer may scientifically conclude from its output.
MAT-X is an AI-assisted engineering investigation system. It is not a certification
system and does not replace qualified engineering judgment or laboratory testing.
## 2. Material identity
The identity service provides candidate-based identification from available evidence.
A candidate is not a guaranteed material-grade determination. Confidence or ranking
fields must be interpreted according to the actual response schema and evidence supplied.
## 3. Process history
Process and heat-treatment records represent available stored evidence. Missing process
steps must not be inferred as though they were observed.
## 4. Property predictions
The backend exposes supported property-prediction operations, including tensile-strength
and yield-strength predictions. A prediction is a model output, not a laboratory
measurement.
The API may expose model-version and applicability information. The actual response
schema is defined by `openapi.json`.
### Uncertainty
A calibrated prediction interval must not be claimed unless a calibration method and
compatible calibration artifact have been implemented and validated.
Applicability-distance or quantile-distance information is not automatically a residual
prediction interval. Input completeness is not equivalent to training-data coverage.
When the evidence is insufficient, the system should communicate that limitation.
## 5. Microstructure analysis
The microstructure service accepts image evidence and provides supported analysis,
map, feature, and provenance operations.
Exploratory segmentation or extracted image features must not be described as validated
phase identification, grain-boundary characterization, or universal metallurgical
classification unless those capabilities have been separately validated for the
specific image domain and labels.
## 6. Damage and condition assessment
The damage service may organize or assess available damage evidence. It must not be
presented as a guaranteed diagnosis of failure mechanism, severity, or remaining useful
life.
The current implementation explicitly limits claims around diagnosis, severity
assessment, and remaining-life prediction. Consult actual API response fields.
## 7. Active investigation
The investigation engine recommends potential next steps based on available evidence
and the implemented recommendation method.
Where a recommendation is heuristic or based on supplied probability distributions,
it must not be described as experimentally validated information gain. A recommendation
is decision support for a human investigator, not an instruction to perform an unsafe
test.
## 8. Model validation
The Version 1.0 research specification identifies these validation designs:
- Random train/test split.
- Grade-held-out split.
- Family-held-out split where feasible.
- Source-held-out split.
- Leakage checks for duplicate specimens, derived target leakage, and publication
  identifiers.
The existence of backend endpoints or tests does not prove that all scientific
experiments have been completed. Experimental results should be reported separately.
## 9. Responsible-use boundaries
MAT-X must not claim:
- Guaranteed failure diagnosis.
- Guaranteed remaining useful life.
- Certified engineering decisions.
- Replacement for laboratory testing.
- Replacement for qualified materials engineers.
- Universal applicability outside the validated training domain.
## 10. Interpretation checklist
Before relying on a result, ask:
1. Is the value observed, computational, derived, imputed, or predicted?
2. What source and record support it?
3. What model version generated the prediction?
4. Is the input inside the model's applicability domain?
5. Is uncertainty actually calibrated, or is it another diagnostic quantity?
6. What evidence is missing?
7. Does the conclusion require additional laboratory characterization?
""")
    write_doc("CONFIGURATION_AND_OPERATIONS.md", """
# Configuration and Operations
## 1. Application entry point
The FastAPI application entry point is:
`app.main:app`
The API router is mounted under:
`/api/v1`
## 2. Startup
```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
For local development with automatic reload:
```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
```
## 3. Application lifecycle
Application startup initializes the MAT-X database and evidence registry. Initialization
and migration behavior is implemented in the backend database and evidence services.
Do not manually delete database files to resolve a migration problem. Back up the
database and inspect migration logs first.
## 4. Environment and secrets
Review the repository's environment template and configuration modules to identify
the exact required variables. This documentation intentionally does not invent
environment-variable names.
Keep secrets in backend environment configuration. Never expose private keys or
database credentials to the browser.
## 5. Logs and troubleshooting
When an endpoint fails, record:
- Request method and path.
- HTTP status code.
- Sanitized request identifier, if available.
- Error response body.
- Backend traceback from the server log.
- Whether the failure came from validation, persistence, a model, or an external source.
Do not include API keys, passwords, private user data, or uploaded confidential files
in bug reports.
## 6. CORS
Configure allowed frontend origins explicitly when the deployment origin is known.
Do not enable unrestricted wildcard origins as a production shortcut.
## 7. Database backup
Before migrations or deployment:
1. Stop writes where practical.
2. Back up the primary database and evidence registry.
3. Preserve uploaded evidence files.
4. Record the application version and migration version.
5. Verify restoration procedures.
## 8. Deployment boundary
Deployment is a separate phase from local backend development. Before production,
confirm the production database, evidence-file persistence, environment variables,
CORS origins, network access, health monitoring, logging, backup, and secret management.
""")
    write_doc("FRONTEND_HANDOFF.md", """
# MAT-X Frontend Integration Handoff
## 1. Backend API base
Local development:
`http://127.0.0.1:8000/api/v1`
The production base URL must be supplied by the backend deployment owner after deployment.
Do not assume a production URL before it is confirmed.
## 2. Contract sources
Use these backend-generated contract files:
- `openapi.json` — machine-readable OpenAPI schema.
- `API_REFERENCE.md` — human-readable endpoint inventory.
- `/docs` — interactive Swagger UI while the backend is running.
These describe the registered routes and schemas. They take precedence over conceptual
endpoint names in planning documents.
## 3. Integration rules
- Use the exact HTTP method and path shown in OpenAPI.
- Match request-body fields and data types exactly.
- Match query/path parameter names and required flags.
- Handle the actual response envelope; do not assume every endpoint returns the same shape.
- Handle validation errors and non-2xx responses.
- Display unavailable or insufficient evidence explicitly.
- Preserve the difference between observed and predicted values.
- Display model version and provenance when returned.
- Do not manufacture missing uncertainty intervals or confidence scores.
- Do not assume an external dataset search always succeeds.
- Keep API keys and database credentials on the backend.
## 4. Suggested frontend API client
The following is a generic TypeScript example. Replace the base URL with the configured
backend URL and use generated types from the actual OpenAPI contract.
```typescript
const API_BASE_URL =
  process.env.NEXT_PUBLIC_MATX_API_BASE_URL ?? "http://127.0.0.1:8000";
export async function apiRequest<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      body?.detail
        ? JSON.stringify(body.detail)
        : `MAT-X API request failed: ${response.status}`
    );
  }
  return body as T;
}
```
For multipart file-upload routes, do not manually set the `Content-Type` header.
Allow the browser to set the multipart boundary.
## 5. API integration sequence
A recommended integration sequence is:
1. Health and model metadata.
2. Material identity and scientific-record operations.
3. Process history and material-state reconstruction.
4. Evidence upload and evidence retrieval.
5. Microstructure analysis and feature retrieval.
6. Property predictions and prediction history.
7. Damage/condition evidence.
8. Investigation recommendation and persistence.
9. Engineering report generation.
10. Dataset search and provenance views.
The exact supported operations and request fields must be checked in `API_REFERENCE.md`.
## 6. Error handling
The frontend should distinguish:
- Client validation errors.
- Missing resources.
- Unsupported operations or material families.
- Insufficient evidence.
- External dataset provider failures.
- Internal server errors.
Do not convert an error or unavailable result into a scientifically meaningful zero.
## 7. Frontend acceptance checklist
- [ ] Frontend can reach the configured API base URL.
- [ ] Health endpoint returns the documented response.
- [ ] OpenAPI schema loads successfully.
- [ ] Requests match backend Pydantic schemas.
- [ ] Successful responses render correctly.
- [ ] Validation errors are displayed clearly.
- [ ] File uploads use multipart form data where required.
- [ ] Long-running operations show loading states.
- [ ] Missing evidence remains visibly missing.
- [ ] Experimental and predicted values are visually distinct.
- [ ] Provenance and model-version fields are retained.
- [ ] No backend secrets are included in frontend bundles.
""")
    write_doc("SPECIFICATION_TRACEABILITY.md", """
# MAT-X Version 1.0 Specification Traceability
## 1. Purpose
This matrix maps the Version 1.0 project specification to the backend capabilities
identified during development. It distinguishes an implemented software capability
from a scientifically validated research result.
**Status meanings**
- **Implemented:** A corresponding backend route or service exists.
- **Partial / bounded:** A capability exists but has a documented scientific or
  implementation limitation.
- **External / data-dependent:** Depends on public provider availability or datasets.
- **Research validation pending:** Requires separate experimental evidence; passing
  software tests alone does not establish scientific validation.
## 2. Functional traceability
| Version 1.0 requirement | Backend implementation evidence | Status |
|---|---|---|
| Material identity | Material identity candidate operation | Implemented; candidate is not guaranteed identity |
| Process history | Process-history and heat/grade retrieval routes | Implemented |
| Microstructure analysis | Image analysis, maps, features, unified map, provenance routes | Partial / exploratory |
| Property reconstruction | Tensile-strength, yield-strength, and inclusion prediction routes | Implemented for supported operations |
| Damage / condition | Damage assessment and evidence operations | Partial / bounded |
| Active investigation | Investigation recommendation and persistence routes | Implemented; method limitations apply |
| Material-state reconstruction | Material state reconstruction route and service | Implemented with evidence limitations |
| Engineering report | Engineering report route | Implemented |
| Evidence upload and retrieval | Evidence registry and file routes | Implemented |
| Dataset adapters | AFLOW, JARVIS, SteelBench, Zenodo, and local dataset routes | Implemented; provider/data dependent |
| Model version metadata | Model-version migration and prediction persistence integration | Implemented |
| Health and model metadata | Health and model routes | Implemented |
| Scientific-record persistence | Generic scientific-record CRUD and lifecycle routes | Implemented |
## 3. API contract note
The specification lists conceptual service paths such as `/materials`, `/predictions`,
`/uncertainty`, and `/next-measurement`. The current backend exposes its implemented
versioned route set as shown in the generated `API_REFERENCE.md`.
Frontend developers must integrate with the actual OpenAPI paths. A conceptual
specification path must not be treated as an implemented alias unless it appears in
OpenAPI.
## 4. Scientific requirements and boundaries
| Specification requirement | Documentation position |
|---|---|
| Preserve provenance | Implemented through source/evidence and record metadata where available |
| Separate experimental/computational/derived/imputed/predicted | Required data interpretation rule; verify individual record metadata |
| Prediction uncertainty | Expose supported uncertainty/applicability fields; do not claim calibrated intervals without calibration evidence |
| Applicability domain | Use implemented warnings and applicability outputs; avoid overclaiming |
| Damage diagnosis | Not claimed as validated diagnosis |
| Remaining useful life | Not claimed |
| Microstructure phase/grain identification | Do not claim universal validated segmentation or metallurgical identification |
| Information gain | Recommendation method must be described according to its actual implementation and validation |
| Grade/source-held-out generalization | Research validation must be evidenced separately from API tests |
| Experiments A-G | Treat as research evaluation plan/results, not automatically completed by backend tests |
## 5. Research experiments
The specification identifies:
- A: Baseline property prediction.
- B: Composition plus heat treatment.
- C: Composition plus heat treatment plus microstructure features.
- D: Multimodal model.
- E: Uncertainty estimation.
- F: Grade-held-out validation.
- G: Active-investigation simulation.
The completion of backend routes does not, by itself, prove that these experiments have
all been executed or passed. Record datasets, split strategies, metrics, model versions,
and experiment outputs in the research report.
## 6. Final interpretation
The backend provides the software foundation for the MAT-X Version 1.0 investigation
workflow. External dataset limitations and research-validation work should be tracked
separately from the availability of backend API operations.
""")
    write_doc("TESTING.md", """
# MAT-X Backend Testing and Verification
## 1. Run the complete test suite
From the repository root:
```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```
## 2. Run a focused test module
```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests\\ -q
```
To run one test file:
```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests\\test_model_versions_schema.py -q
```
Replace the file name with the test module being investigated.
## 3. API schema verification
Start the application:
```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
```
Then inspect:
- `/docs`
- `/openapi.json`
The documentation generator also imports `app.main:app` and exports its OpenAPI schema.
## 4. Previously reported regression baseline
The latest user-provided test output before this documentation generation was:
- **463 passed**
- **5 warnings**
- Runtime approximately 50.24 seconds
This is historical evidence from the development session, not a claim that the current
checkout has just been retested. The PowerShell handoff command runs the current suite
and stores its output in `TEST_RESULTS_LATEST.txt`.
## 5. Warnings
Warnings should be reviewed rather than hidden. Previously observed warnings included
a Starlette/httpx deprecation warning and scikit-learn feature-name warnings.
A passing test suite verifies the tested software behavior. It does not establish
scientific model calibration, external dataset completeness, or engineering certification.
""")
    schema = generate_openapi()
    generate_api_reference(schema)
    generate_database_inventory()
    # Repository inventory helps maintainers locate implementation modules.
    app_files = relative_files("app", {".py"})
    tests = relative_files("tests", {".py"})
    write_doc("CODE_INVENTORY.md", "\n".join([
        "# Backend Code Inventory",
        "",
        f"> Generated on {TODAY}.",
        "",
        "## Application Python files",
        "",
        *([f"- `{x}`" for x in app_files] or ["No `app/` Python files discovered."]),
        "",
        "## Test files",
        "",
        *([f"- `{x}`" for x in tests] or ["No `tests/` Python files discovered."]),
        "",
        "## Maintenance note",
        "",
        "This inventory is generated from the current checkout. It is not a substitute "
        "for reading service implementations, schemas, migrations, and tests.",
        "",
    ]))
    print("\nDOCUMENTATION GENERATION FINISHED")
    print(f"Documentation directory: {DOCS}")
    print(f"OpenAPI generated: {'YES' if schema else 'NO'}")
    print(f"Python application files inventoried: {len(app_files)}")
    print(f"Test files inventoried: {len(tests)}")
if __name__ == "__main__":
    main()