# MAT-X Backend — Version 1.0
AI-Assisted Materials State Reconstruction and Active Investigation Engine.
## Requirements
- Python 3.12
- Dependencies listed in `requirements.txt`
- MAT-X model artifacts in the `models/` directory
- Scientific datasets supplied separately where required
## Local setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, storage
$env:MATX_DATASET_DIR = (Resolve-Path .\data).Path
$env:MATX_STORAGE_DIR = (Resolve-Path .\storage).Path
```
Set `MATX_DATASET_DIR` and `MATX_STORAGE_DIR` in the environment before starting.
The default dataset directory in the application is the user's Downloads/MatX_Dataset
folder; configure it explicitly when using a different location.
Start the API:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
## API
- Base path: `/api/v1`
- Interactive OpenAPI documentation: `/docs`
- OpenAPI schema: `/openapi.json`
- Root service check: `/`
The application initializes its evidence registry and MAT-X database during
FastAPI lifespan startup.
## Persistent storage and deployment
Set `MATX_STORAGE_DIR` to a persistent writable directory. It contains the
SQLite database and uploaded evidence files. Do not use ephemeral container
storage for production data.
Set `MATX_DATASET_DIR` to a mounted directory containing the required datasets.
The Docker image intentionally excludes local datasets and application storage.
Mount both directories when running the container.
Build:
```bash
docker build -t matx-backend:1.0 .
```
Run with persistent host directories:
```bash
docker run --rm -p 8000:8000 \
  -e MATX_STORAGE_DIR=/app/storage \
  -e MATX_DATASET_DIR=/app/data \
  -v matx-storage:/app/storage \
  -v matx-data:/app/data \
  matx-backend:1.0
```
Supply the required scientific datasets in the mounted data volume before using
endpoints that depend on them. Preserve model artifacts from `models/`.
## Important scientific limitations
- Prediction intervals remain unavailable unless a compatible calibration
  artifact is developed, validated, and promoted.
- Applicability-domain results are not calibrated prediction intervals.
- Explainability output is global model-level feature importance, not
  per-prediction or causal explanation.
- Engineering outputs require qualified engineering review and are not
  certified assessments.
## Release verification
Run the backend regression suite:
```bash
python -m pytest -q
```
Check installed dependency consistency:
```bash
python -m pip check
```

