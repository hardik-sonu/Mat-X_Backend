# Backend Code Inventory

> Generated on 2026-11-15T07:38:34+05:00.

## Application Python files

- `app/__init__.py`
- `app/api/__init__.py`
- `app/api/routes.py`
- `app/core/__init__.py`
- `app/core/characterization.py`
- `app/core/model_registry.py`
- `app/core/paths.py`
- `app/engines/active_investigation_engine.py`
- `app/engines/active_investigation_experiment_evidence.py`
- `app/engines/applicability_engine.py`
- `app/engines/damage_condition_engine.py`
- `app/engines/ebsd_provenance_engine.py`
- `app/engines/fatigue_engine.py`
- `app/engines/fatigue_response_engine.py`
- `app/engines/material_identity_engine.py`
- `app/engines/microstructure_feature_engine.py`
- `app/engines/process_history_analysis.py`
- `app/engines/process_history_engine.py`
- `app/engines/process_history_validation.py`
- `app/integrations/__init__.py`
- `app/integrations/aflow_service.py`
- `app/integrations/jarvis_service.py`
- `app/integrations/local_dataset_service.py`
- `app/integrations/steelbench_service.py`
- `app/integrations/zenodo_service.py`
- `app/main.py`
- `app/schemas/__init__.py`
- `app/schemas/aflow.py`
- `app/schemas/api_responses.py`
- `app/schemas/damage_condition.py`
- `app/schemas/evidence.py`
- `app/schemas/investigation.py`
- `app/schemas/jarvis.py`
- `app/schemas/local_dataset.py`
- `app/schemas/material_identity.py`
- `app/schemas/material_state.py`
- `app/schemas/prediction_responses.py`
- `app/schemas/predictions.py`
- `app/schemas/steelbench.py`
- `app/schemas/zenodo.py`
- `app/services/__init__.py`
- `app/services/ebsd_provenance_service.py`
- `app/services/engineering_report_service.py`
- `app/services/evidence_file_service.py`
- `app/services/evidence_registry_service.py`
- `app/services/material_identity_service.py`
- `app/services/material_state_service.py`
- `app/services/matx_database_service.py`
- `app/services/matx_repository_service.py`
- `app/services/micrograph_analysis_service.py`
- `app/services/microstructure_service.py`
- `app/services/prediction_persistence_service.py`
- `app/services/prediction_service.py`
- `app/services/process_history_service.py`

## Test files

- `tests/_evidence_smoke_test.py`
- `tests/test_active_investigation_engine.py`
- `tests/test_active_investigation_experiment_evidence.py`
- `tests/test_aflow_api.py`
- `tests/test_applicability_api.py`
- `tests/test_applicability_engine.py`
- `tests/test_damage_condition_engine.py`
- `tests/test_data_lake_paths.py`
- `tests/test_engine_api.py`
- `tests/test_engineering_report_api.py`
- `tests/test_evidence_api.py`
- `tests/test_evidence_file_service.py`
- `tests/test_evidence_registry.py`
- `tests/test_evidence_xlsx_upload.py`
- `tests/test_experiment_g_hide_reveal.py`
- `tests/test_fatigue_response_engine.py`
- `tests/test_jarvis_api.py`
- `tests/test_local_dataset_integration.py`
- `tests/test_material_state_api.py`
- `tests/test_matx_database.py`
- `tests/test_matx_lifecycle.py`
- `tests/test_matx_repository.py`
- `tests/test_matx_repository_api.py`
- `tests/test_micrograph_image_api.py`
- `tests/test_microstructure_api.py`
- `tests/test_microstructure_feature_engine.py`
- `tests/test_model_versions_schema.py`
- `tests/test_prediction_explainability.py`
- `tests/test_prediction_input_evidence.py`
- `tests/test_prediction_persistence.py`
- `tests/test_predictive_uncertainty_contract.py`
- `tests/test_process_history_service.py`
- `tests/test_process_history_validation.py`
- `tests/test_steelbench_integration.py`
- `tests/test_steelbench_manifest.py`
- `tests/test_v1_api_traceability.py`
- `tests/test_zenodo_integration.py`

## Maintenance note

This inventory is generated from the current checkout. It is not a substitute for reading service implementations, schemas, migrations, and tests.
