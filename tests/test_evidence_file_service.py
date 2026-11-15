import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile
from app.services.evidence_file_service import store_evidence_upload
def make_upload(filename, content):
    return UploadFile(filename=filename, file=BytesIO(content))
def test_valid_csv_upload_and_checksum(tmp_path):
    content = b"material_id,hardness\nMATX-001,220\n"
    result = asyncio.run(
        store_evidence_upload(
            make_upload("hardness.csv", content),
            tmp_path,
        )
    )
    stored = Path(result["storage_path"])
    assert stored.read_bytes() == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["media_type"] == "text/csv"
def test_windows_path_filename_is_sanitized(tmp_path):
    result = asyncio.run(
        store_evidence_upload(
            make_upload(r"..\unsafe\hardness.csv", b"a,b\n1,2\n"),
            tmp_path,
        )
    )
    assert result["original_filename"] == "hardness.csv"
    assert Path(result["storage_path"]).parent == tmp_path.resolve()
def test_invalid_utf8_is_rejected_and_cleaned(tmp_path):
    with pytest.raises(ValueError, match="UTF-8"):
        asyncio.run(
            store_evidence_upload(
                make_upload("bad.csv", b"valid,text\n\xff"),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
def test_nul_byte_anywhere_in_text_file_is_rejected(tmp_path):
    content = b"a" * 100 + b"\x00" + b"b,c\n"
    with pytest.raises(ValueError, match="Binary content"):
        asyncio.run(
            store_evidence_upload(
                make_upload("binary.csv", content),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
def test_mismatched_pdf_signature_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            store_evidence_upload(
                make_upload("fake.pdf", b"not a PDF"),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Empty"):
        asyncio.run(
            store_evidence_upload(
                make_upload("empty.txt", b""),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
def test_unsupported_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        asyncio.run(
            store_evidence_upload(
                make_upload("program.exe", b"content"),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
def test_utf8_character_at_header_boundary_is_accepted(tmp_path):
    content = b"a" * 15 + "é".encode("utf-8") + b"\n"
    result = asyncio.run(
        store_evidence_upload(
            make_upload("unicode.txt", content),
            tmp_path,
        )
    )
    assert Path(result["storage_path"]).read_bytes() == content
def test_upload_over_25_mib_is_rejected_and_cleaned(tmp_path):
    from app.services.evidence_file_service import MAX_UPLOAD_BYTES
    content = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="25 MiB"):
        asyncio.run(
            store_evidence_upload(
                make_upload("large.txt", content),
                tmp_path,
            )
        )
    assert list(tmp_path.iterdir()) == []
import asyncio
from io import BytesIO
from pathlib import Path
from starlette.datastructures import UploadFile
from app.services.evidence_file_service import store_evidence_upload
def test_supported_binary_file_signatures(tmp_path):
    samples = {
        "sample.pdf": b"%PDF-1.7\nMAT-X test",
        "sample.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        "sample.jpg": b"\xff\xd8\xff" + b"\x00" * 16,
        "sample.tif": b"II*\x00" + b"\x00" * 16,
    }
    for filename, content in samples.items():
        result = asyncio.run(
            store_evidence_upload(
                UploadFile(filename=filename, file=BytesIO(content)),
                tmp_path,
            )
        )
        stored = Path(result["storage_path"])
        assert stored.read_bytes() == content
        assert result["original_filename"] == filename
def test_invalid_tiff_signature_is_rejected(tmp_path):
    try:
        asyncio.run(
            store_evidence_upload(
                UploadFile(
                    filename="invalid.tiff",
                    file=BytesIO(b"NOT TIFF CONTENT"),
                ),
                tmp_path,
            )
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Invalid TIFF signature was accepted.")
    assert list(tmp_path.iterdir()) == []
