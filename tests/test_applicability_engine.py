import pytest
from app.engines.applicability_engine import (
    assess_applicability,
    load_reference,
)
TARGETS = [
    "tensile_strength",
    "yield_strength",
]
def make_complete_payload(target):
    artifact = load_reference(target)
    payload = {
        feature: artifact["reference_statistics"]["numeric"][feature]["median"]
        for feature in artifact["features"]["numeric"]
    }
    for feature in artifact["features"]["categorical"]:
        payload[feature] = sorted(
            artifact["reference_statistics"]["categorical"][feature]
        )[0]
    return payload
@pytest.mark.parametrize("target", TARGETS)
def test_complete_input_returns_valid_screening(target):
    result = assess_applicability(target, make_complete_payload(target))
    assert result["status"] in {
        "IN_DOMAIN",
        "BORDERLINE",
        "OUT_OF_DOMAIN",
    }
    assert result["distance"] is not None
    assert result["provided_feature_count"] == 13
    assert result["missing_features"] == []
    assert result["confidence_probability"] is None
@pytest.mark.parametrize("target", TARGETS)
def test_missing_input_returns_insufficient_evidence(target):
    payload = make_complete_payload(target)
    payload.pop("C")
    result = assess_applicability(target, payload)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["distance"] is None
    assert "C" in result["missing_features"]
@pytest.mark.parametrize("target", TARGETS)
def test_unseen_category_is_flagged(target):
    payload = make_complete_payload(target)
    payload["quench_medium"] = "MATX_UNSEEN_CATEGORY"
    result = assess_applicability(target, payload)
    assert "quench_medium" in result["unseen_categories"]
    assert result["status"] in {"BORDERLINE", "OUT_OF_DOMAIN"}
@pytest.mark.parametrize("target", TARGETS)
def test_extreme_numeric_value_is_out_of_domain(target):
    payload = make_complete_payload(target)
    payload["C"] = 1_000_000.0
    result = assess_applicability(target, payload)
    assert result["status"] == "OUT_OF_DOMAIN"
    assert result["distance"] is not None
    assert any(
        item["feature"] == "C"
        for item in result["range_violations"]
    )
@pytest.mark.parametrize("target", TARGETS)
def test_non_finite_numeric_input_is_rejected(target):
    payload = make_complete_payload(target)
    payload["C"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assess_applicability(target, payload)
def test_unsupported_target_is_rejected():
    with pytest.raises(ValueError, match="Unsupported applicability target"):
        assess_applicability("inclusion_classifier", {})
@pytest.mark.parametrize("target", TARGETS)
def test_missing_categorical_input_returns_insufficient_evidence(target):
    payload = make_complete_payload(target)
    payload["steel_family"] = None
    result = assess_applicability(target, payload)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert "steel_family" in result["missing_features"]
    assert result["distance"] is None
def test_resolve_promoted_model_path_handles_windows_provenance(monkeypatch, tmp_path):
    from app.engines import applicability_engine
    model_file = tmp_path / "steelbench_tensile_extratrees.joblib"
    model_file.write_bytes(b"test-model")
    monkeypatch.setattr(applicability_engine, "MODEL_DIR", tmp_path)
    provenance = {
        "promoted_model": (
            "C:/Users/Ahmadcomputer/MAT-X/models/"
            "steelbench_tensile_extratrees.joblib"
        )
    }
    resolved = applicability_engine._resolve_promoted_model_path(provenance)
    assert resolved == model_file
    assert resolved.is_file()
def test_resolve_promoted_model_path_rejects_missing_model(monkeypatch, tmp_path):
    from app.engines import applicability_engine
    monkeypatch.setattr(applicability_engine, "MODEL_DIR", tmp_path)
    provenance = {
        "promoted_model": (
            r"C:\Users\Ahmadcomputer\MAT-X\models\missing_model.joblib"
        )
    }
    with pytest.raises(FileNotFoundError, match="models directory"):
        applicability_engine._resolve_promoted_model_path(provenance)
def test_resolve_promoted_model_path_rejects_empty_provenance():
    from app.engines import applicability_engine
    with pytest.raises(ValueError, match="invalid promoted model path"):
        applicability_engine._resolve_promoted_model_path({})

