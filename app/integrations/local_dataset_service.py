"""Safe, read-only discovery and querying of local tabular datasets."""
import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any
from app.core.paths import DATASET_ROOT, PROJECT_ROOT
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_BYTES = 50 * 1024 * 1024
class LocalDatasetError(RuntimeError):
    pass
def _dataset_roots() -> list[Path]:
    """Return configured and packaged dataset locations in priority order."""
    roots = [
        Path(DATASET_ROOT).expanduser().resolve(),
        (
            Path(PROJECT_ROOT)
            / "data"
            / "raw"
            / "steelbench"
        ).resolve(),
    ]
    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots
def _resolve_dataset(filename: str) -> Path:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A dataset filename is required.")
    if filename != Path(filename).name or filename in {".", ".."}:
        raise ValueError("Only filenames inside the dataset directory are allowed.")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only CSV and XLSX datasets are supported.")
    for root in _dataset_roots():
        path = (root / filename).resolve()
        if path.parent != root:
            raise ValueError("Dataset path must remain inside the dataset directory.")
        if path.is_file():
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Dataset exceeds the 50 MB processing limit.")
            return path
    raise FileNotFoundError("The requested local dataset does not exist.")
def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise LocalDatasetError("CSV dataset has no header.")
        columns = [str(name).strip() if name is not None else "" for name in reader.fieldnames]
        if any(not name for name in columns):
            raise LocalDatasetError("CSV dataset contains an empty column name.")
        if len(columns) != len(set(columns)):
            raise LocalDatasetError("CSV dataset contains duplicate column names.")
        records = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise LocalDatasetError(
                    f"CSV row {line_number} contains extra fields."
                )
            if all(value is None or not str(value).strip() for value in row.values()):
                continue
            records.append({
                key: _clean_value(value)
                for key, value in row.items()
            })
        return columns, records
def _read_xlsx(path: Path):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise LocalDatasetError(
            "XLSX support requires the openpyxl package."
        ) from exc
    try:
        workbook = load_workbook(
            path,
            read_only=True,
            data_only=True,
        )
        worksheet = workbook.active
        iterator = worksheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if not header:
            raise LocalDatasetError("XLSX dataset has no header.")
        columns = [
            str(value).strip() if value is not None else ""
            for value in header
        ]
        if any(not name for name in columns):
            raise LocalDatasetError("XLSX dataset contains an empty column name.")
        if len(columns) != len(set(columns)):
            raise LocalDatasetError("XLSX dataset contains duplicate column names.")
        records = []
        for row in iterator:
            if all(value is None for value in row):
                continue
            if len(row) > len(columns) and any(
                value is not None for value in row[len(columns):]
            ):
                raise LocalDatasetError("XLSX dataset contains extra fields.")
            values = list(row[:len(columns)])
            values.extend([None] * (len(columns) - len(values)))
            records.append({
                column: _clean_value(value)
                for column, value in zip(columns, values)
            })
        return columns, records
    except LocalDatasetError:
        raise
    except Exception as exc:
        raise LocalDatasetError(
            f"XLSX dataset could not be read: {type(exc).__name__}."
        ) from exc
    finally:
        if "workbook" in locals():
            workbook.close()
def list_local_datasets() -> dict:
    selected_root = Path(DATASET_ROOT).expanduser().resolve()
    datasets = []
    # Prefer the configured directory when it contains supported datasets.
    # Otherwise discover datasets packaged with the application.
    for root in _dataset_roots():
        if not root.is_dir():
            continue
        candidates = []
        for path in sorted(root.iterdir()):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            candidates.append({
                "filename": path.name,
                "format": path.suffix.lower().lstrip(".").upper(),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime
                ).astimezone().isoformat(),
            })
        if candidates:
            selected_root = root
            datasets = candidates
            break
    return {
        "status": "SUCCESS",
        "dataset_directory": str(selected_root),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }

def search_local_dataset_records(
    *,
    filename: str,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    if isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    if isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be zero or greater.")
    if query is not None and len(query) > 500:
        raise ValueError("query must not exceed 500 characters.")
    path = _resolve_dataset(filename)
    try:
        if path.suffix.lower() == ".csv":
            columns, records = _read_csv(path)
        else:
            columns, records = _read_xlsx(path)
    except UnicodeDecodeError as exc:
        raise LocalDatasetError(
            "CSV encoding is unsupported; use UTF-8."
        ) from exc
    except csv.Error as exc:
        raise LocalDatasetError("CSV parsing failed.") from exc
    matches = records
    if query and query.strip():
        term = query.strip().casefold()
        matches = [
            record for record in records
            if any(
                term in str(value).casefold()
                for value in record.values()
                if value is not None
            )
        ]
    selected = matches[offset:offset + limit]
    return {
        "status": "SUCCESS",
        "dataset": filename,
        "format": path.suffix.lower().lstrip(".").upper(),
        "columns": columns,
        "total_source_records": len(records),
        "total_matches": len(matches),
        "record_count": len(selected),
        "limit": limit,
        "offset": offset,
        "records": selected,
        "source_provenance": {
            "classification": "LOCAL_DATA_AS_PROVIDED",
            "source_filename": filename,
            "source_sha256": _sha256(path),
        },
        "scientific_notice": (
            "Records are returned as supplied by the local file. "
            "No scientific validation, unit harmonization, or "
            "measurement certification is performed."
        ),
    }
def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
