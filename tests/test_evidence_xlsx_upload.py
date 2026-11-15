import asyncio
import hashlib
import io
import zipfile
import pytest
from starlette.datastructures import UploadFile
from app.services.evidence_file_service import store_evidence_upload
def make_xlsx_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '</Types>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheets/></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"/>',
        )
    return buffer.getvalue()
def test_valid_xlsx_upload_is_stored_and_hashed(tmp_path):
    content = make_xlsx_bytes()
    upload = UploadFile(
        filename="measurements.xlsx",
        file=io.BytesIO(content),
    )
    result = asyncio.run(store_evidence_upload(upload, tmp_path))
    assert result["original_filename"] == "measurements.xlsx"
    assert result["stored_filename"].endswith(".xlsx")
    assert result["media_type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["size_bytes"] == len(content)
    assert (tmp_path / result["stored_filename"]).read_bytes() == content
def test_xlsx_extension_with_non_zip_content_is_rejected(tmp_path):
    upload = UploadFile(
        filename="measurements.xlsx",
        file=io.BytesIO(b"This is not an XLSX workbook."),
    )
    with pytest.raises(ValueError, match="content does not match"):
        asyncio.run(store_evidence_upload(upload, tmp_path))
    assert list(tmp_path.iterdir()) == []
def test_xlsx_zip_without_workbook_parts_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unrelated.txt", "not a workbook")
    upload = UploadFile(
        filename="measurements.xlsx",
        file=io.BytesIO(buffer.getvalue()),
    )
    with pytest.raises(ValueError, match="missing required components"):
        asyncio.run(store_evidence_upload(upload, tmp_path))
    assert list(tmp_path.iterdir()) == []
def test_xlsx_with_malformed_workbook_xml_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types>")
        archive.writestr("xl/workbook.xml", "<workbook>")
        archive.writestr("xl/_rels/workbook.xml.rels", "<Relationships/>")
    upload = UploadFile(
        filename="measurements.xlsx",
        file=io.BytesIO(buffer.getvalue()),
    )
    with pytest.raises(ValueError, match="invalid XML"):
        asyncio.run(store_evidence_upload(upload, tmp_path))
    assert list(tmp_path.iterdir()) == []
