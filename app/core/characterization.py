CHARACTERIZATION_CATEGORIES = (
    "chemical_composition",
    "microstructure",
    "phase_identification",
    "grain_characterization",
    "mechanical_testing",
    "process_history",
    "damage_assessment",
)
def validate_uncertainty_categories(value: dict) -> dict:
    """Reject uncertainty entries outside the supported characterization contract."""
    unknown = set(value) - set(CHARACTERIZATION_CATEGORIES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported uncertainty categories: {names}.")
    return value
