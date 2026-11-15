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
cd C:\Users\Ahmadcomputer\MAT-X
```
## 3. Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use the interpreter directly:
```powershell
.\.venv\Scripts\python.exe --version
```
## 4. Install dependencies
Use the dependency file that exists in the repository:
```powershell
if (Test-Path requirements.txt) {
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
elseif (Test-Path pyproject.toml) {
    .\.venv\Scripts\python.exe -m pip install -e .
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
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
Default local address:
`http://127.0.0.1:8000`
## 7. Verify the service
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health endpoint: consult the generated API reference for the exact path and response.
## 8. Run tests
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
## 9. Regenerate documentation
```powershell
.\.venv\Scripts\python.exe scripts\generate_backend_docs.py
```
Run the generator from the project root so that imports and relative paths resolve.
