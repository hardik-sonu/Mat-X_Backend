import numpy as np
import pandas as pd
import pytest
from training.experiments.experiment_g_hide_reveal import (
    build_evidence_conditions,
    calculate_metrics,
    paired_bootstrap_ci,
)
def test_build_evidence_conditions_preserves_source_frame():
    frame = pd.DataFrame({
        "C": [0.2, 0.3],
        "Mn": [0.5, 0.6],
        "Cr": [0.1, 0.2],
        "Mo": [0.01, 0.02],
        "Ni": [0.1, 0.2],
        "Si": [0.2, 0.3],
        "V": [0.01, 0.02],
        "Cu": [0.1, 0.2],
        "Al": [0.01, 0.02],
        "austenitize_T": [850, 900],
        "temper_T": [550, 600],
        "quench_medium": ["oil", "water"],
        "steel_family": ["alloy", "alloy"],
    })
    original = frame.copy(deep=True)
    conditions = build_evidence_conditions(frame)
    assert set(conditions) == {
        "hidden_baseline",
        "composition_only",
        "process_only",
        "full_evidence",
    }
    assert conditions["hidden_baseline"]["C"].isna().all()
    assert conditions["hidden_baseline"]["austenitize_T"].isna().all()
    assert conditions["composition_only"]["C"].equals(frame["C"])
    assert conditions["composition_only"]["austenitize_T"].isna().all()
    assert conditions["process_only"]["C"].isna().all()
    assert conditions["process_only"]["austenitize_T"].equals(
        frame["austenitize_T"]
    )
    assert conditions["full_evidence"].equals(frame)
    pd.testing.assert_frame_equal(frame, original)
def test_build_evidence_conditions_requires_all_group_features():
    with pytest.raises(ValueError, match="Missing required"):
        build_evidence_conditions(pd.DataFrame({"C": [0.2]}))
def test_calculate_metrics_returns_mae_and_rmse():
    result = calculate_metrics(
        np.array([10.0, 20.0]),
        np.array([12.0, 17.0]),
    )
    assert result["mae"] == pytest.approx(2.5)
    assert result["rmse"] == pytest.approx(np.sqrt(6.5))
def test_calculate_metrics_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics(
            np.array([10.0, np.nan]),
            np.array([10.0, 12.0]),
        )
def test_paired_bootstrap_ci_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = paired_bootstrap_ci(values, seed=42, replicates=500)
    second = paired_bootstrap_ci(values, seed=42, replicates=500)
    assert first == second
    assert first["mean"] == pytest.approx(2.5)
    assert first["ci_95_lower"] <= first["mean"]
    assert first["ci_95_upper"] >= first["mean"]
def test_paired_bootstrap_ci_rejects_invalid_input():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([]))
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, 2.0]), replicates=10)