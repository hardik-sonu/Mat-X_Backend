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
