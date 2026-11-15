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
