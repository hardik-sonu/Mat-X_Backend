import csv
import hashlib
import json
from pathlib import Path
def test_steelbench_manifest_matches_data_lake_dataset():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "metadata" / "steelbench_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = root / manifest["relative_path"]
    assert dataset_path.is_file()
    assert manifest["dataset_id"] == "steelbench_v1"
    assert manifest["provenance"]["classification"] == "SOURCE_DATA_AS_PUBLISHED"
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest().upper()
    assert manifest["sha256"] == digest
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == manifest["columns"]
    assert len(rows) == manifest["record_count"]
    assert len(manifest["columns"]) == manifest["column_count"]
def test_steelbench_manifest_declares_source_values():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data" / "metadata" / "steelbench_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["source_field"] == "source"
    assert manifest["source_values"]
