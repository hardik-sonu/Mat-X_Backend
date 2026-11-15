# Configuration and Operations
## 1. Application entry point
The FastAPI application entry point is:
`app.main:app`
The API router is mounted under:
`/api/v1`
## 2. Startup
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
For local development with automatic reload:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
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
