
from __future__ import annotations
import base64
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP"}
def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    probabilities = histogram / total
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (global_mean * cumulative_probability - cumulative_mean) ** 2
    valid = denominator > 0
    if not np.any(valid):
        raise ValueError("Image has insufficient grayscale contrast.")
    scores = np.zeros(256, dtype=np.float64)
    scores[valid] = numerator[valid] / denominator[valid]
    return int(np.argmax(scores))
def _region_statistics(mask: np.ndarray, scale: float | None) -> dict:
    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    labels, count = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sizes[sizes >= 3]
    if sizes.size == 0:
        return {
            "component_count": 0,
            "area_pixels_mean": None,
            "area_pixels_median": None,
            "area_pixels_p90": None,
            "equivalent_diameter_pixels_mean": None,
            "equivalent_diameter_um_mean": None,
        }
    diameters = np.sqrt(4.0 * sizes.astype(float) / math.pi)
    return {
        "component_count": int(sizes.size),
        "area_pixels_mean": float(np.mean(sizes)),
        "area_pixels_median": float(np.median(sizes)),
        "area_pixels_p90": float(np.percentile(sizes, 90)),
        "equivalent_diameter_pixels_mean": float(np.mean(diameters)),
        "equivalent_diameter_um_mean": (
            float(np.mean(diameters) * scale)
            if scale is not None else None
        ),
    }
def analyze_micrograph_image(
    image_bytes: bytes,
    filename: str,
    scale_um_per_pixel: float | None = None,
) -> dict:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("An image file is required.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MB upload limit.")
    if scale_um_per_pixel is not None:
        if (
            isinstance(scale_um_per_pixel, bool)
            or not math.isfinite(float(scale_um_per_pixel))
            or float(scale_um_per_pixel) <= 0
        ):
            raise ValueError("Pixel scale must be finite and positive.")
        scale_um_per_pixel = float(scale_um_per_pixel)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format.")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 25-megapixel limit.")
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Uploaded file is not a valid readable image.") from exc
    if gray.size == 0 or int(gray.min()) == int(gray.max()):
        raise ValueError("Image has insufficient grayscale contrast.")
    threshold = _otsu_threshold(gray)
    dark_mask = gray <= threshold
    light_mask = ~dark_mask
    dark_fraction = float(np.mean(dark_mask))
    light_fraction = float(np.mean(light_mask))
    if dark_fraction == 0 or light_fraction == 0:
        raise ValueError("Image could not be segmented into two intensity regions.")
    # Analyze the less-populated intensity class as an exploratory feature mask.
    feature_mask = dark_mask if dark_fraction <= light_fraction else light_mask
    mask_buffer = io.BytesIO()
    Image.fromarray(feature_mask.astype(np.uint8) * 255).save(
        mask_buffer, format="PNG"
    )
    return {
        "analysis_status": "EXPLORATORY_IMAGE_FEATURES",
        "source_filename": Path(filename or "uploaded-image").name,
        "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "source_format": image_format,
        "image_width_pixels": int(width),
        "image_height_pixels": int(height),
        "grayscale_mean": float(np.mean(gray)),
        "grayscale_std": float(np.std(gray)),
        "grayscale_min": int(np.min(gray)),
        "grayscale_max": int(np.max(gray)),
        "segmentation": {
            "method": "OTSU_GLOBAL_BINARY_INTENSITY_THRESHOLD",
            "threshold": threshold,
            "dark_region_fraction": dark_fraction,
            "light_region_fraction": light_fraction,
            "analyzed_region": (
                "DARK_INTENSITY" if dark_fraction <= light_fraction
                else "LIGHT_INTENSITY"
            ),
            "mask_format": "PNG_BASE64",
            "mask_base64": base64.b64encode(
                mask_buffer.getvalue()
            ).decode("ascii"),
        },
        "morphology_features": _region_statistics(
            feature_mask, scale_um_per_pixel
        ),
        "pixel_scale_um_per_pixel": scale_um_per_pixel,
        "scientific_scope": {
            "phase_identification_performed": False,
            "grain_boundary_identification_validated": False,
            "segmentation_is_exploratory": True,
            "physical_scale_assumed": scale_um_per_pixel is not None,
            "engineer_review_required": True,
        },
        "limitations": [
            "Global intensity segmentation is exploratory.",
            "Connected components are not validated metallurgical grains.",
            "Intensity regions are not identified phases or inclusions.",
            "Physical dimensions require a supplied calibrated pixel scale.",
            "Results require expert review before engineering interpretation.",
        ],
    }
