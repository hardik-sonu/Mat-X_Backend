"""Centralized MAT-X dataset and application storage paths."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
def resolve_steelbench_path(project_root: Path, dataset_root: Path) -> Path:
    """Prefer the standardized data lake, retaining legacy compatibility."""
    data_lake_path = (
        Path(project_root)
        / "data"
        / "raw"
        / "steelbench"
        / "SteelBench v1.0.csv"
    )
    legacy_path = Path(dataset_root) / "SteelBench v1.0.csv"
    if data_lake_path.is_file():
        return data_lake_path.resolve()
    return legacy_path.resolve()
# Existing shared root remains available to EBSD and process-history services.
DATASET_ROOT = Path(
    os.getenv(
        "MATX_DATASET_DIR",
        str(Path.home() / "Downloads" / "MatX_Dataset"),
    )
).expanduser().resolve()
# SteelBench prefers its standardized data-lake location.
STEELBENCH_PATH = resolve_steelbench_path(PROJECT_ROOT, DATASET_ROOT)
# Application storage is separate from original scientific datasets.
_DEFAULT_STORAGE_ROOT = Path.home() / "MATX_Storage"
STORAGE_ROOT = Path(
    os.getenv("MATX_STORAGE_DIR", str(_DEFAULT_STORAGE_ROOT))
).expanduser().resolve()
EVIDENCE_FILES_DIR = STORAGE_ROOT / "evidence"
EVIDENCE_DB_PATH = STORAGE_ROOT / "matx_evidence.sqlite3"
