# MAT-X Backend Documentation
**Project:** MAT-X — AI-Assisted Materials State Reconstruction and Active Investigation Engine  
**Specification baseline:** Project Specification, Version 1.0  
**Documentation generated:** 2026-11-15T07:38:34+05:00
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
.\.venv\Scripts\Activate.ps1
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
.\.venv\Scripts\python.exe scripts\generate_backend_docs.py
```
Review the generated files before committing or sharing them.
