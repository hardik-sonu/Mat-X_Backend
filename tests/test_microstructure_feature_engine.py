import hashlib
import numpy as np
import pytest
from app.engines.microstructure_feature_engine import (
    _sha256_file,
    _validate_scan_geometry,
)
def test_sha256_file_streams_and_matches_standard_digest(tmp_path):
    path = tmp_path / "large-map.h5oina"
    content = b"MAT-X EBSD test data" * 100000
    path.write_bytes(content)
    assert _sha256_file(path, chunk_size=4096) == hashlib.sha256(
        content
    ).hexdigest()
def test_valid_ebsd_scan_geometry_is_accepted():
    area = _validate_scan_geometry(1800, 1200, 1.0, 1.0, 2160000)
    assert area == 2160000.0
@pytest.mark.parametrize(
    "x_cells,y_cells,x_step,y_step,total_points",
    [
        (0, 1200, 1.0, 1.0, 0),
        (1800, -1, 1.0, 1.0, 2160000),
        (1800, 1200, 0.0, 1.0, 2160000),
        (1800, 1200, 1.0, -1.0, 2160000),
        (1800, 1200, float("nan"), 1.0, 2160000),
        (1800, 1200, 1.0, float("inf"), 2160000),
        (1800, 1200, 1.0, 1.0, 100),
    ],
    ids=[
        "zero-x-dimension",
        "negative-y-dimension",
        "zero-x-step",
        "negative-y-step",
        "nan-x-step",
        "infinite-y-step",
        "point-count-mismatch",
    ],
)
def test_invalid_ebsd_scan_geometry_is_rejected(
    x_cells, y_cells, x_step, y_step, total_points
):
    with pytest.raises(ValueError):
        _validate_scan_geometry(
            x_cells, y_cells, x_step, y_step, total_points
        )
