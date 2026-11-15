
import base64
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.services.micrograph_analysis_service import analyze_micrograph_image
client = TestClient(app)
def make_test_image():
    pixels = np.full((128, 128), 220, dtype=np.uint8)
    pixels[15:20, :] = 30
    pixels[45:50, :] = 30
    pixels[75:80, :] = 30
    pixels[105:110, :] = 30
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()
def test_micrograph_service_extracts_exploratory_features():
    result = analyze_micrograph_image(
        make_test_image(), "synthetic-micrograph.png", 0.5
    )
    assert result["analysis_status"] == "EXPLORATORY_IMAGE_FEATURES"
    assert result["image_width_pixels"] == 128
    assert result["image_height_pixels"] == 128
    assert result["source_sha256"]
    assert result["morphology_features"]["component_count"] >= 1
    assert result["morphology_features"]["equivalent_diameter_um_mean"] is not None
    assert result["scientific_scope"]["phase_identification_performed"] is False
def test_micrograph_service_does_not_invent_physical_scale():
    result = analyze_micrograph_image(make_test_image(), "unscaled.png")
    assert result["pixel_scale_um_per_pixel"] is None
    assert result["morphology_features"]["equivalent_diameter_um_mean"] is None
    assert result["scientific_scope"]["physical_scale_assumed"] is False
def test_micrograph_service_rejects_invalid_image():
    with pytest.raises(ValueError, match="valid readable image"):
        analyze_micrograph_image(b"not-an-image", "broken.png")
def test_micrograph_service_rejects_invalid_pixel_scale():
    with pytest.raises(ValueError, match="finite and positive"):
        analyze_micrograph_image(make_test_image(), "sample.png", 0)
def test_micrograph_api_success_returns_features_and_mask():
    response = client.post(
        "/api/v1/microstructure/analyze",
        files={"file": ("sample.png", make_test_image(), "image/png")},
        data={"scale_um_per_pixel": "0.5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "EXPLORATORY_IMAGE_FEATURES"
    assert body["source_sha256"]
    assert body["segmentation"]["mask_format"] == "PNG_BASE64"
    mask = base64.b64decode(body["segmentation"]["mask_base64"])
    with Image.open(io.BytesIO(mask)) as image:
        assert image.size == (128, 128)
def test_micrograph_api_rejects_malformed_upload():
    response = client.post(
        "/api/v1/microstructure/analyze",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 422
def test_micrograph_api_rejects_invalid_scale():
    response = client.post(
        "/api/v1/microstructure/analyze",
        files={"file": ("sample.png", make_test_image(), "image/png")},
        data={"scale_um_per_pixel": "0"},
    )
    assert response.status_code == 422
