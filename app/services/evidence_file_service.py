from __future__ import annotations
import codecs
import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote
import httpx
from fastapi import UploadFile
from app.core.database import database_backend
from app.core.paths import EVIDENCE_FILES_DIR
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

CHUNK_SIZE = 1024 * 1024
def _supabase_storage_config() -> tuple[str, str, str]:
    """Return validated private Supabase Storage configuration."""
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    secret_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "matx-evidence").strip()
    if not base_url or not secret_key or not bucket:
        raise RuntimeError(
            "PostgreSQL evidence storage requires SUPABASE_URL, "
            "SUPABASE_SECRET_KEY, and SUPABASE_STORAGE_BUCKET."
        )
    if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise RuntimeError("SUPABASE_URL must use HTTPS.")
    if any(ord(char) < 32 for char in base_url + bucket):
        raise RuntimeError("Supabase Storage configuration contains invalid characters.")
    return base_url, secret_key, bucket
def is_remote_evidence_storage() -> bool:
    """Whether the configured application backend requires remote evidence storage."""
    return database_backend() == "postgres"
def _supabase_object_url(base_url: str, bucket: str, filename: str) -> str:
    return (
        f"{base_url}/storage/v1/object/"
        f"{quote(bucket, safe='')}/{quote(filename, safe='')}"
    )
def download_stored_evidence_file(stored_filename: str) -> bytes:
    """Download one private Supabase object using the server-side secret key."""
    if (
        not isinstance(stored_filename, str)
        or not stored_filename
        or Path(stored_filename).name != stored_filename
        or "/" in stored_filename
        or "\\" in stored_filename
    ):
        raise ValueError("Invalid stored evidence filename.")
    base_url, secret_key, bucket = _supabase_storage_config()
    url = (
        f"{base_url}/storage/v1/object/authenticated/"
        f"{quote(bucket, safe='')}/{quote(stored_filename, safe='')}"
    )
    headers = {
        "Authorization": f"Bearer {secret_key}",
        "apikey": secret_key,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            if response.status_code == 404:
                raise FileNotFoundError("Evidence object was not found.")
            response.raise_for_status()
            return response.content
    except FileNotFoundError:
        raise
    except httpx.HTTPError as exc:
        raise RuntimeError("Supabase evidence download failed.") from exc
ALLOWED_TYPES = {'.pdf': 'application/pdf',
 '.png': 'image/png',
 '.jpg': 'image/jpeg',
 '.jpeg': 'image/jpeg',
 '.tif': 'image/tiff',
 '.tiff': 'image/tiff',
 '.csv': 'text/csv',
 '.txt': 'text/plain',
 '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
def _safe_original_filename(filename: str | None) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("An uploaded filename is required.")
    # Normalize both Windows and POSIX path separators.
    leaf = filename.replace("\\", "/").split("/")[-1]
    # Remove control characters and surrounding whitespace.
    leaf = "".join(
        char for char in leaf
        if ord(char) >= 32 and ord(char) != 127
    ).strip()
    if not leaf or len(leaf) > 255:
        raise ValueError("The original filename is invalid.")
    return leaf
def _validate_content(
    extension: str,
    header: bytes,
    text_decoder: Any = None,
) -> None:
    signatures = {
        ".xlsx": lambda data: data.startswith(b"PK\x03\x04"),
        ".pdf": lambda data: data.startswith(b"%PDF-"),
        ".png": lambda data: data.startswith(
            b"\x89PNG\r\n\x1a\n"
        ),
        ".jpg": lambda data: data.startswith(b"\xff\xd8\xff"),
        ".jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
        ".tif": lambda data: data.startswith(
            (b"II*\x00", b"MM\x00*")
        ),
        ".tiff": lambda data: data.startswith(
            (b"II*\x00", b"MM\x00*")
        ),
    }
    if extension in signatures:
        if not signatures[extension](header):
            raise ValueError(
                "File content does not match its declared extension."
            )
        return
    if extension in {".csv", ".txt"}:
        if b"\x00" in header:
            raise ValueError("Binary content is not allowed for text files.")
        # Full-file UTF-8 validity is checked incrementally during upload.
        return
    raise ValueError("Unsupported evidence file type.")

def _validate_xlsx_file(path: Path) -> None:
    """Validate an XLSX ZIP container and its essential workbook XML."""
    import xml.etree.ElementTree as ET
    import zipfile
    required_parts = {
        "[Content_Types].xml",
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.infolist()
            if len(entries) > 10000:
                raise ValueError("XLSX archive contains too many entries.")
            total_uncompressed = sum(item.file_size for item in entries)
            if total_uncompressed > 100 * 1024 * 1024:
                raise ValueError(
                    "XLSX uncompressed content exceeds the 100 MiB limit."
                )
            names = set()
            for item in entries:
                name = item.filename
                parts = name.split("/")
                if (
                    name.startswith("/")
                    or "\\" in name
                    or ".." in parts
                    or (item.flag_bits & 0x1)
                ):
                    raise ValueError(
                        "XLSX archive contains an unsafe or encrypted entry."
                    )
                names.add(name)
            missing = required_parts - names
            if missing:
                raise ValueError(
                    "XLSX workbook is missing required components: "
                    + ", ".join(sorted(missing))
                )
            for part in required_parts:
                try:
                    ET.fromstring(archive.read(part))
                except ET.ParseError as exc:
                    raise ValueError(
                        f"XLSX workbook contains invalid XML: {part}"
                    ) from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("XLSX file is not a valid ZIP workbook.") from exc
    except OSError as exc:
        raise ValueError("XLSX workbook could not be read.") from exc

async def store_evidence_upload(
    upload_file: UploadFile,
    storage_dir: Path | str | None = None,
) -> dict:
    """Store one validated evidence file using a generated filename."""
    original_filename = _safe_original_filename(upload_file.filename)
    extension = Path(original_filename).suffix.lower()
    if extension not in ALLOWED_TYPES:
        raise ValueError("Unsupported evidence file extension.")
    # Explicit directories are always local (including isolated test fixtures).
    # PostgreSQL deployments use a temporary local staging directory before
    # transferring the validated object to private Supabase Storage.
    remote_storage = storage_dir is None and database_backend() == "postgres"
    if remote_storage:
        _supabase_storage_config()  # Fail before accepting data if configuration is incomplete.
        storage_root = (
            Path(tempfile.gettempdir()) / "MATX_Evidence_Uploads"
        ).resolve()
    else:
        storage_root = Path(
            storage_dir if storage_dir is not None else EVIDENCE_FILES_DIR
        ).expanduser().resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    generated_name = f"{uuid.uuid4().hex}{extension}"
    final_path = storage_root / generated_name
    temporary_path = storage_root / f".{generated_name}.uploading"
    digest = hashlib.sha256()
    total_bytes = 0
    header = bytearray()
    text_decoder = (
        codecs.getincrementaldecoder("utf-8")()
        if extension in {".csv", ".txt"}
        else None
    )
    promoted = False
    try:
        with temporary_path.open("xb") as destination:
            while True:
                chunk = await upload_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        "Evidence file exceeds the 25 MiB upload limit."
                    )
                if len(header) < 16:
                    header.extend(chunk[:16 - len(header)])
                if extension in {".csv", ".txt"} and b"\x00" in chunk:
                    raise ValueError(
                        "Binary content is not allowed for text files."
                    )
                if text_decoder is not None:
                    try:
                        text_decoder.decode(chunk)
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            "CSV and TXT uploads must contain valid UTF-8."
                        ) from exc
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if total_bytes == 0:
            raise ValueError("Empty evidence files are not accepted.")
        if text_decoder is not None:
            try:
                text_decoder.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "CSV and TXT uploads must contain valid UTF-8."
                ) from exc
        _validate_content(extension, bytes(header), text_decoder)
        if extension == ".xlsx":
            _validate_xlsx_file(temporary_path)
        # Rename within the same directory. Generated names prevent
        # user-controlled paths and avoid replacing existing evidence.
        if final_path.exists():
            raise RuntimeError("Generated evidence filename collision.")
        os.rename(temporary_path, final_path)
        promoted = True
        if remote_storage:
            base_url, secret_key, bucket = _supabase_storage_config()
            url = _supabase_object_url(base_url, bucket, generated_name)
            headers = {
                "Authorization": f"Bearer {secret_key}",
                "apikey": secret_key,
                "Content-Type": ALLOWED_TYPES[extension],
                "x-upsert": "false",
            }
            try:
                with final_path.open("rb") as source:
                    content = source.read()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        url,
                        content=content,
                        headers=headers,
                    )
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    "Supabase evidence upload failed."
                ) from exc
            final_path.unlink(missing_ok=True)
            promoted = False
            storage_path = f"supabase://{bucket}/{generated_name}"
        else:
            storage_path = str(final_path)
        return {
            "original_filename": original_filename,
            "stored_filename": generated_name,
            "sha256": digest.hexdigest(),
            "media_type": ALLOWED_TYPES[extension],
            "size_bytes": total_bytes,
            "storage_path": storage_path,
        }
    except Exception:
        temporary_path.unlink(missing_ok=True)
        if promoted:
            final_path.unlink(missing_ok=True)
        raise

def delete_stored_evidence_file(
    stored_filename: str,
    storage_dir: Path | str | None = None,
) -> None:
    """Delete only a generated evidence filename inside the evidence directory."""
    if (
        not isinstance(stored_filename, str)
        or not stored_filename
        or Path(stored_filename).name != stored_filename
        or "/" in stored_filename
        or "\\" in stored_filename
    ):
        raise ValueError("Invalid stored evidence filename.")
    if storage_dir is None and database_backend() == "postgres":
        base_url, secret_key, bucket = _supabase_storage_config()
        url = (
            f"{base_url}/storage/v1/object/{quote(bucket, safe='')}"
        )
        headers = {
            "Authorization": f"Bearer {secret_key}",
            "apikey": secret_key,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    "DELETE",
                    url,
                    json={"prefixes": [stored_filename]},
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Supabase evidence deletion failed.") from exc
        return None
    root = Path(
        storage_dir if storage_dir is not None else EVIDENCE_FILES_DIR
    ).expanduser().resolve()
    target = (root / stored_filename).resolve()
    if target.parent != root:
        raise ValueError("Evidence file path is outside the storage directory.")
    target.unlink(missing_ok=True)
    return None
