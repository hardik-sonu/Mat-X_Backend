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
