# MAT-X Backend Testing and Verification
## 1. Run the complete test suite
From the repository root:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
## 2. Run a focused test module
```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```
To run one test file:
```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_model_versions_schema.py -q
```
Replace the file name with the test module being investigated.
## 3. API schema verification
Start the application:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
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
