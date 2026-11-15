import hashlib
import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, Response

from app.schemas.investigation import ActiveInvestigationInput
from app.schemas.material_identity import MaterialIdentityInput
from app.schemas.material_state import MaterialStateReconstructionInput
from app.schemas.damage_condition import DamageAssessmentInput

from app.schemas.api_responses import (
    DamageAssessmentResponse,
    MaterialStateResponse,
    EngineeringReportResponse,
    HealthResponse,
    ModelsResponse,
    InclusionPredictionResponse,
    MicrostructureMapsResponse,
    FatigueResponse,
    ActiveInvestigationResponse,
    MaterialIdentityResponse,
    ProcessHistoryResponse,
    ProcessHistoryAnalysisResponse,
    ProcessLookupResponse,
    EbsdProvenanceResponse,
    UnifiedMicrostructureResponse,
    EvidenceRecordResponse,
)

from app.engines.fatigue_response_engine import analyze_fatigue_response
from app.engines.damage_condition_engine import assess_damage_condition
from app.engines.active_investigation_engine import recommend_next_investigation
from app.core.model_registry import get_model_status
from app.schemas.predictions import (
    TensilePredictionInput,
    InclusionPredictionInput,
    YieldPredictionInput
)
from app.schemas.prediction_responses import (
    TensilePredictionResponse,
    YieldPredictionResponse,
)
from app.services.material_identity_service import identify_material
from app.services.material_state_service import reconstruct_material_state
from app.services.engineering_report_service import generate_engineering_report
from app.services.prediction_persistence_service import persist_prediction
from app.services.prediction_service import (
    predict_tensile,
    predict_inclusion,
    predict_yield
)
from app.services.microstructure_service import (
    list_microstructure_maps,
    get_microstructure_features,
    get_unified_microstructure_record,
)
from app.services.process_history_service import get_process_history
from app.services.process_history_service import get_process_history_analysis

from app.services.process_history_service import get_heat_process_history, get_grade_process_history

from app.services.ebsd_provenance_service import get_ebsd_provenance

from app.schemas.evidence import EvidenceCreateInput
from app.services.evidence_registry_service import (
    create_evidence_record,
    get_evidence_record,
    list_evidence_records,
)
from app.core.paths import EVIDENCE_FILES_DIR
from app.services.evidence_file_service import (
    store_evidence_upload,
    delete_stored_evidence_file,
    download_stored_evidence_file,
    is_remote_evidence_storage,
)


from app.services.matx_repository_service import (
    MatxRepositoryError,
    create_scientific_record as repository_create_record,
    get_scientific_record as repository_get_record,
    list_scientific_records as repository_list_records,
    update_lifecycle_record as repository_update_lifecycle,
    persist_investigation_recommendations as repository_persist_investigation,
)
router = APIRouter()

@router.post("/damage/assess", response_model=DamageAssessmentResponse)
def damage_condition_assessment(request: DamageAssessmentInput):
    try:
        records = list_evidence_records(
            material_id=request.material_id,
            limit=1000,
        )
        return assess_damage_condition(
            material_id=request.material_id,
            records=records,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.post("/material/state/reconstruct", response_model=MaterialStateResponse)
def material_state_reconstruction(
    request: MaterialStateReconstructionInput
):
    try:
        return reconstruct_material_state(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/engineering/report", response_model=EngineeringReportResponse)
def engineering_report(request: MaterialStateReconstructionInput):
    try:
        return generate_engineering_report(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/health", response_model=HealthResponse)
def health():
    statuses = get_model_status()
    available = sum(
        1 for item in statuses.values()
        if item["status"] == "available"
    )
    return {
        "status": "healthy" if available == len(statuses) else "degraded",
        "service": "MAT-X API",
        "models_available": available,
        "models_registered": len(statuses),
        "models": statuses
    }
@router.get("/models", response_model=ModelsResponse)
def models():
    return {
        "registered_models": get_model_status()
    }
@router.post("/predict/tensile-strength", response_model=TensilePredictionResponse, responses={503: {"description": "Prediction model or persistence service unavailable"}})
def tensile_strength(request: TensilePredictionInput):
    try:
        result = predict_tensile(request)
        result["persistence"] = persist_prediction(
            material_id=request.material_id,
            target_property="tensile_strength",
            prediction_type="NUMERIC",
            request_features=request.model_dump(
                exclude={"material_id"}, exclude_none=True
            ),
            result=result,
            model_key="tensile_strength",
        )
        return result
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.post("/predict/yield-strength", response_model=YieldPredictionResponse, responses={503: {"description": "Prediction model or persistence service unavailable"}})
def yield_strength(request: YieldPredictionInput):
    try:
        result = predict_yield(request)
        result["persistence"] = persist_prediction(
            material_id=request.material_id,
            target_property="yield_strength",
            prediction_type="NUMERIC",
            request_features=request.model_dump(
                exclude={"material_id"}, exclude_none=True
            ),
            result=result,
            model_key="yield_strength",
        )
        return result
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.post("/predict/inclusion", response_model=InclusionPredictionResponse, responses={503: {"description": "Prediction model or persistence service unavailable"}})
def inclusion(request: InclusionPredictionInput):
    try:
        result = predict_inclusion(request)
        result["persistence"] = persist_prediction(
            material_id=request.material_id,
            target_property="inclusion_classification",
            prediction_type="CLASSIFICATION",
            request_features=request.model_dump(
                exclude={"material_id"}, exclude_none=True
            ),
            result=result,
            model_key="inclusion_classifier",
        )
        return result
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.get("/microstructure/maps", response_model=MicrostructureMapsResponse)
def microstructure_maps():
    try:
        return list_microstructure_maps()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/microstructure/maps/{map_id}/features", response_model=dict[str, Any])
def microstructure_features(map_id: str):
    try:
        return get_microstructure_features(map_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/fatigue/response", response_model=FatigueResponse)
def fatigue_response():
    try:
        return analyze_fatigue_response(write_outputs=False, verbose=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/investigation/recommend", response_model=ActiveInvestigationResponse)
def investigation_recommendation(request: ActiveInvestigationInput):
    try:
        return recommend_next_investigation(
            material_id=request.material_id,
            material_family=request.material_family,
            evidence=request.evidence,
            uncertainty=request.uncertainty,
            information_gain_inputs=request.information_gain_inputs,
            investigation_goal=request.investigation_goal,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post(
    "/investigation/recommend/persist",
    response_model=ActiveInvestigationResponse,
)
def persist_investigation_recommendation(request: ActiveInvestigationInput):
    """Recommend and persist proposed experiments without claiming execution."""
    try:
        result = recommend_next_investigation(
            material_id=request.material_id,
            material_family=request.material_family,
            evidence=request.evidence,
            uncertainty=request.uncertainty,
            information_gain_inputs=request.information_gain_inputs,
            investigation_goal=request.investigation_goal,
            top_k=request.top_k,
        )
        persistence = repository_persist_investigation(
            material_id=request.material_id,
            investigation_goal=request.investigation_goal,
            initial_state={
                "material_family": request.material_family,
                "evidence": request.evidence,
                "uncertainty": request.uncertainty,
                "information_gain_inputs": request.information_gain_inputs,
                "engine": result["engine"],
                "engine_version": result["engine_version"],
                "evidence_summary": result["evidence_summary"],
                "methodology": result["methodology"],
            },
            limitations=result["limitations"],
            recommendations=result["recommendations"],
        )
        return {**result, "persistence": persistence}
    except MatxRepositoryError as exc:
        status_code = 404 if str(exc).startswith("Material not found:") else 422
        raise HTTPException(status_code=status_code, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.post("/material/identity/candidates", response_model=MaterialIdentityResponse)
def material_identity_candidates(request: MaterialIdentityInput):
    try:
        return identify_material(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/process/history", response_model=ProcessHistoryResponse)
def process_history():
    try:
        return get_process_history()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/process/history/analysis", response_model=ProcessHistoryAnalysisResponse)
def process_history_analysis():
    try:
        return get_process_history_analysis()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/process/history/heat/{heat_id}", response_model=ProcessLookupResponse)
def process_history_by_heat(heat_id: str):
    try:
        result = get_heat_process_history(heat_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No process history found for heat ID: {heat_id}"
            )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
@router.get("/process/history/grade/{grade_id}", response_model=ProcessLookupResponse)
def process_history_by_grade(grade_id: str):
    try:
        result = get_grade_process_history(grade_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No process history found for grade ID: {grade_id}"
            )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
@router.get("/microstructure/provenance", response_model=EbsdProvenanceResponse)
def microstructure_provenance():
    try:
        return get_ebsd_provenance()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@router.get("/microstructure/maps/{map_id}/unified", response_model=UnifiedMicrostructureResponse)
def unified_microstructure_record(map_id: str):
    try:
        return get_unified_microstructure_record(map_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


from app.integrations.jarvis_service import (
    search_jarvis_structures,
    JarvisIntegrationError,
)
from app.schemas.jarvis import JarvisSearchResponse
@router.get(
    "/datasets/jarvis/structures",
    response_model=JarvisSearchResponse,
)
def jarvis_structures(
    elements: list[str] = Query(
        ...,
        description="Chemical element symbols, e.g. Fe and Ni.",
    ),
    page_limit: int = Query(default=10, ge=1, le=100),
):
    try:
        return search_jarvis_structures(
            elements=elements,
            page_limit=page_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except JarvisIntegrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
from app.integrations.steelbench_service import (
    search_steelbench_records,
    SteelBenchIntegrationError,
)
from app.schemas.steelbench import SteelBenchSearchResponse
from app.integrations.local_dataset_service import (
    list_local_datasets,
    search_local_dataset_records,
    LocalDatasetError,
)
from app.schemas.local_dataset import (
    LocalDatasetListResponse,
    LocalDatasetSearchResponse,
)
from app.integrations.zenodo_service import (
    search_zenodo_records,
    ZenodoIntegrationError,
)
from app.schemas.zenodo import ZenodoSearchResponse
from app.integrations.aflow_service import (
    search_aflow_structures,
    AflowIntegrationError,
)
from app.schemas.aflow import AflowSearchResponse
@router.get(
    "/datasets/aflow/search",
    response_model=AflowSearchResponse,
)
def aflow_search(
    elements: list[str] = Query(
        ...,
        description="Chemical element symbols, e.g. Fe and Ni.",
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    exact_species_count: int | None = Query(
        default=None,
        ge=1,
        le=7,
    ),
    require_properties: bool = Query(default=True),
):
    try:
        return search_aflow_structures(
            elements=elements,
            page=page,
            page_size=page_size,
            exact_species_count=exact_species_count,
            require_properties=require_properties,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AflowIntegrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get(
    "/datasets/steelbench/records",
    response_model=SteelBenchSearchResponse,
)
def steelbench_records(
    heat_id: str | None = Query(default=None, min_length=1, max_length=128),
    grade_id: str | None = Query(default=None, min_length=1, max_length=128),
    steel_family: str | None = Query(default=None, min_length=1, max_length=128),
    source: str | None = Query(default=None, min_length=1, max_length=128),
    data_tier: str | None = Query(default=None, min_length=1, max_length=128),
    query: str | None = Query(default=None, min_length=1, max_length=128),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    try:
        return search_steelbench_records(
            heat_id=heat_id,
            grade_id=grade_id,
            steel_family=steel_family,
            source=source,
            data_tier=data_tier,
            query=query,
            limit=limit,
            offset=offset,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except SteelBenchIntegrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="SteelBench dataset request failed.")
@router.get(
    "/datasets/zenodo/search",
    response_model=ZenodoSearchResponse,
)
def zenodo_search(
    query: str = Query(..., min_length=1, max_length=500),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=25),
    sort: str = Query(default="bestmatch"),
):
    try:
        return search_zenodo_records(
            query=query,
            page=page,
            size=size,
            sort=sort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ZenodoIntegrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
@router.get(
    "/datasets/local",
    response_model=LocalDatasetListResponse,
)
def local_dataset_list():
    return list_local_datasets()
@router.get(
    "/datasets/local/records",
    response_model=LocalDatasetSearchResponse,
)
def local_dataset_search(
    filename: str = Query(..., min_length=1, max_length=255),
    query: str | None = Query(default=None, max_length=500),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    try:
        return search_local_dataset_records(
            filename=filename,
            query=query,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LocalDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.post("/evidence", status_code=201, response_model=EvidenceRecordResponse)
def create_evidence(request: EvidenceCreateInput):
    """Create an immutable scientific evidence record."""
    try:
        payload = request.model_dump(exclude_unset=True)
        return create_evidence_record(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.get("/evidence", response_model=list[EvidenceRecordResponse])
def list_evidence(
    material_id: str | None = Query(default=None, min_length=1),
    investigation_id: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """List evidence records with optional filters."""
    try:
        return list_evidence_records(
            material_id=material_id,
            investigation_id=investigation_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.get("/evidence/{evidence_id}", response_model=EvidenceRecordResponse)
def retrieve_evidence(evidence_id: str):
    """Retrieve one evidence record by its identifier."""
    try:
        record = get_evidence_record(evidence_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Evidence record not found.",
            )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@router.post("/evidence/upload", status_code=201, response_model=EvidenceRecordResponse)
async def upload_evidence(
    metadata: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a scientific evidence file and register its provenance."""
    try:
        metadata_payload = json.loads(metadata)
        if not isinstance(metadata_payload, dict):
            raise ValueError("Metadata must be a JSON object.")
        reserved_fields = {
            "original_filename",
            "stored_filename",
            "sha256",
            "media_type",
        }
        supplied_reserved = reserved_fields.intersection(metadata_payload)
        if supplied_reserved:
            raise ValueError(
                "File metadata fields are generated by the server: "
                + ", ".join(sorted(supplied_reserved))
            )
        evidence_request = EvidenceCreateInput.model_validate(
            metadata_payload
        )
        payload = evidence_request.model_dump(exclude_unset=True)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        stored = await store_evidence_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    payload.update({
        "original_filename": stored["original_filename"],
        "stored_filename": stored["stored_filename"],
        "sha256": stored["sha256"],
        "media_type": stored["media_type"],
    })
    try:
        return create_evidence_record(payload)
    except ValueError as exc:
        delete_stored_evidence_file(stored["stored_filename"])
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        delete_stored_evidence_file(stored["stored_filename"])
        raise HTTPException(
            status_code=500,
            detail="Evidence file could not be registered.",
        )

@router.get(
    "/evidence/{evidence_id}/file",
    response_class=FileResponse,
    responses={
        200: {
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "description": "Verified evidence file download.",
        }
    },
)
def download_evidence_file(evidence_id: str):
    """Download evidence after storage-path and SHA-256 verification."""
    record = get_evidence_record(evidence_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Evidence record not found.",
        )
    stored_filename = record.get("stored_filename")
    if (
        not isinstance(stored_filename, str)
        or not stored_filename
        or Path(stored_filename).name != stored_filename
        or "/" in stored_filename
        or "\\" in stored_filename
    ):
        raise HTTPException(
            status_code=404,
            detail="Evidence file is not available.",
        )
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise HTTPException(
            status_code=409,
            detail="Evidence file checksum is unavailable.",
        )
    if is_remote_evidence_storage():
        try:
            content = download_stored_evidence_file(stored_filename)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Evidence file is not available.",
            )
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Evidence storage service is unavailable.",
            )
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise HTTPException(
                status_code=409,
                detail="Evidence file checksum verification failed.",
            )
        original_filename = record.get("original_filename")
        if isinstance(original_filename, str):
            original_filename = original_filename.replace("\\", "/").split("/")[-1]
            original_filename = "".join(
                char for char in original_filename
                if ord(char) >= 32 and ord(char) != 127
            ).strip()
        else:
            original_filename = None
        download_name = original_filename or stored_filename
        from urllib.parse import quote
        disposition = (
            "attachment; filename*=UTF-8''"
            + quote(download_name, safe="")
        )
        return Response(
            content=content,
            media_type=record.get("media_type") or "application/octet-stream",
            headers={"Content-Disposition": disposition},
        )
    root = Path(EVIDENCE_FILES_DIR).expanduser().resolve()
    candidate = (root / stored_filename).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence file is not available.",
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise HTTPException(
            status_code=409,
            detail="Evidence file checksum verification failed.",
        )
    original_filename = record.get("original_filename")
    if isinstance(original_filename, str):
        original_filename = original_filename.replace("\\", "/").split("/")[-1]
        original_filename = "".join(
            char for char in original_filename
            if ord(char) >= 32 and ord(char) != 127
        ).strip()
    else:
        original_filename = None
    download_name = original_filename or stored_filename
    return FileResponse(
        path=candidate,
        media_type=record.get("media_type") or "application/octet-stream",
        filename=download_name,
    )
# ---------------------------------------------------------------------------
# Scientific persistence API
# Evidence records intentionally remain in their dedicated immutable API.
# ---------------------------------------------------------------------------
@router.post(
    "/scientific-records/{table}",
    status_code=201,
    response_model=dict[str, Any],
)
def create_scientific_record_api(
    table: str,
    payload: dict[str, Any] = Body(...),
):
    """Create an append-only record in an allowed scientific table."""
    try:
        return repository_create_record(table, payload)
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.get(
    "/scientific-records/{table}",
    response_model=list[dict[str, Any]],
)
def list_scientific_records_api(
    table: str,
    filters_json: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List scientific records using optional JSON equality filters."""
    try:
        filters = {}
        if filters_json is not None:
            filters = json.loads(filters_json)
            if not isinstance(filters, dict):
                raise ValueError("filters_json must contain a JSON object.")
        return repository_list_records(
            table,
            filters=filters,
            limit=limit,
            offset=offset,
        )
    except (MatxRepositoryError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
@router.get(
    "/scientific-records/{table}/{record_id}",
    response_model=dict[str, Any],
)
def retrieve_scientific_record_api(table: str, record_id: str):
    """Retrieve a scientific record by its table-specific identifier."""
    try:
        record = repository_get_record(table, record_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Scientific record not found.",
            )
        return record
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

# ---------------------------------------------------------------------------
# Controlled investigation and experiment lifecycle API
# ---------------------------------------------------------------------------
@router.patch(
    "/scientific-records/{table}/{record_id}/lifecycle",
    response_model=dict[str, Any],
)
def update_scientific_lifecycle_api(
    table: str,
    record_id: str,
    payload: dict[str, Any] = Body(...),
):
    """Advance an investigation or experiment through valid lifecycle states."""
    try:
        record = repository_update_lifecycle(table, record_id, payload)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Scientific lifecycle record not found.",
            )
        return record
    except MatxRepositoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

from fastapi import File, Form, UploadFile
from app.services.micrograph_analysis_service import (
    MAX_IMAGE_BYTES,
    analyze_micrograph_image,
)
@router.post("/microstructure/analyze")
async def analyze_microstructure_upload(
    file: UploadFile = File(...),
    scale_um_per_pixel: float | None = Form(default=None),
):
    try:
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
        return analyze_micrograph_image(
            image_bytes=image_bytes,
            filename=file.filename or "uploaded-image",
            scale_um_per_pixel=scale_um_per_pixel,
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Micrograph analysis failed.")
    finally:
        await file.close()

