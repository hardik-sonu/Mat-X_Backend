from pathlib import Path
source = Path("training/regression/run_applicability_domain_assessment.py")
destination = Path("training/regression/run_applicability_domain_assessment_v2.py")
code = source.read_text(encoding="utf-8")
replacements = [
    (
        'OUT = ROOT / "training_results" / "applicability_domain_v1"',
        'OUT = ROOT / "training_results" / "applicability_domain_v2"',
    ),
    (
        "def assess_variant(\n    train,\n    test,",
        "def assess_variant(\n    train,\n    calibration,\n    test,",
    ),
    (
        '''    reference_distances, reference_indices = (
        nearest_other_grade_distances(
            train_numeric,
            train_categories,
            train_grades,
        )
    )
    q95, q99 = np.quantile(
        reference_distances, [0.95, 0.99]
    )
''',
        '''    (
        _,
        calibration_numeric,
        _,
        _,
        _,
    ) = prepare_numeric(
        train, calibration, numeric_features
    )
    (
        _,
        calibration_categories,
        _,
    ) = prepare_categories(
        train, calibration, categorical_features
    )
    calibration_distance_matrix = mixed_distance_matrix(
        calibration_numeric,
        train_numeric,
        calibration_categories,
        train_categories,
        len(numeric_features),
        len(categorical_features),
    )
    calibration_nearest_distances = np.min(
        calibration_distance_matrix, axis=1
    )
    if (
        len(calibration_nearest_distances) == 0
        or not np.isfinite(calibration_nearest_distances).all()
    ):
        raise RuntimeError(
            "Invalid calibration-to-training distances."
        )
    q95, q99 = np.quantile(
        calibration_nearest_distances, [0.95, 0.99]
    )
''',
    ),
    (
        '"training_distance_reference_rows": len(reference_distances),',
        '"calibration_reference_rows": len(calibration_nearest_distances),',
    ),
    (
        '''result, summary, feature_rows = assess_variant(
                        train=train,
                        test=test,''',
        '''result, summary, feature_rows = assess_variant(
                        train=train,
                        calibration=calibration,
                        test=test,''',
    ),
    (
        '''"Each training record's nearest record from a different grade."''',
        '''"Each calibration record's nearest training record."''',
    ),
    (
        '''"95th and 99th percentiles of training cross-grade "
                "nearest-neighbour distances."''',
        '''"95th and 99th percentiles of calibration-to-training "
                "nearest-neighbour distances."''',
    ),
]
for old, new in replacements:
    if old not in code:
        raise RuntimeError(
            "Expected source section not found; refusing partial modification:\n"
            + old[:250]
        )
    code = code.replace(old, new, 1)
if 'applicability_domain_v2' not in code:
    raise RuntimeError("V2 output path was not configured.")
if 'calibration=calibration' not in code:
    raise RuntimeError("Calibration partition was not passed.")
destination.write_text(code, encoding="utf-8")
compile(code, str(destination), "exec")
print("V2 script created and syntax-checked:")
print(destination.resolve())
print("V1 script remains unchanged.")
