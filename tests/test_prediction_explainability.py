from types import SimpleNamespace
import pytest
from app.services.prediction_service import build_model_explainability
def test_global_feature_importance_is_ranked_and_named():
    model = SimpleNamespace(feature_importances_=[0.2, 0.7, 0.1])
    result = build_model_explainability(model, ["carbon", "manganese", "silicon"])
    assert result["status"] == "AVAILABLE"
    assert result["method"] == (
        "EXTRA_TREES_IMPURITY_BASED_GLOBAL_FEATURE_IMPORTANCE"
    )
    assert result["scope"] == "GLOBAL_MODEL_LEVEL_NOT_PER_PREDICTION_OR_CAUSAL"
    assert result["features"][0] == {
        "feature": "manganese",
        "importance": 0.7,
        "rank": 1,
    }
    assert result["features"][1]["feature"] == "carbon"
def test_global_feature_importance_preserves_stable_tie_order():
    model = SimpleNamespace(feature_importances_=[0.5, 0.5])
    result = build_model_explainability(model, ["first", "second"])
    assert [item["feature"] for item in result["features"]] == [
        "first", "second"
    ]
def test_explainability_is_explicitly_unavailable_without_native_importance():
    result = build_model_explainability(object(), ["carbon"])
    assert result["status"] == "NOT_AVAILABLE"
    assert result["features"] == []
def test_explainability_rejects_feature_count_mismatch():
    model = SimpleNamespace(feature_importances_=[0.2, 0.8])
    with pytest.raises(RuntimeError, match="count does not match"):
        build_model_explainability(model, ["carbon"])
def test_explainability_rejects_nonfinite_importance():
    model = SimpleNamespace(feature_importances_=[float("nan")])
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        build_model_explainability(model, ["carbon"])
