from pathlib import Path
from app.core.paths import resolve_steelbench_path
def test_steelbench_prefers_data_lake_copy(tmp_path):
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "legacy"
    data_lake = (
        project_root / "data" / "raw" / "steelbench"
        / "SteelBench v1.0.csv"
    )
    legacy = dataset_root / "SteelBench v1.0.csv"
    data_lake.parent.mkdir(parents=True)
    dataset_root.mkdir(parents=True)
    data_lake.write_text("source,data\\nnew,1\\n", encoding="utf-8")
    legacy.write_text("source,data\\nlegacy,1\\n", encoding="utf-8")
    assert resolve_steelbench_path(project_root, dataset_root) == data_lake.resolve()
def test_steelbench_falls_back_to_legacy_dataset(tmp_path):
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "legacy"
    legacy = dataset_root / "SteelBench v1.0.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("source,data\\nlegacy,1\\n", encoding="utf-8")
    assert resolve_steelbench_path(project_root, dataset_root) == legacy.resolve()
def test_steelbench_returns_legacy_fallback_when_data_lake_is_missing(tmp_path):
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "legacy"
    result = resolve_steelbench_path(project_root, dataset_root)
    assert result == (
        dataset_root / "SteelBench v1.0.csv"
    ).resolve()
