from pathlib import Path
import json
import pandas as pd
import numpy as np
OUT = Path(
    r"C:\Users\Ahmadcomputer\MAT-X\training_results"
) / "microstructure" / "abaqus_deep_audit"
thresholds = [2, 5, 10, 15]
expected_pixels = 73163
rows = []
for threshold in thresholds:
    path = OUT / f"ctf_grain_geometry_{threshold}deg.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "grain_id",
        "pixel_count",
        "area_ctf_coordinate_units_squared",
        "equivalent_diameter_ctf_coordinate_units"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{threshold} degree file missing columns: {missing}"
        )
    pixels = df["pixel_count"].astype(int)
    diameter = df["equivalent_diameter_ctf_coordinate_units"]
    if int(pixels.sum()) != expected_pixels:
        raise RuntimeError(
            f"{threshold} degree pixel conservation failed."
        )
    if df["grain_id"].duplicated().any():
        raise RuntimeError(
            f"{threshold} degree duplicate grain IDs."
        )
    rows.append({
        "threshold_degrees": threshold,
        "grain_count": int(len(df)),
        "indexed_pixels": int(pixels.sum()),
        "mean_pixels_per_grain": float(pixels.mean()),
        "median_pixels": float(pixels.median()),
        "p90_pixels": float(pixels.quantile(.90)),
        "p95_pixels": float(pixels.quantile(.95)),
        "maximum_pixels": int(pixels.max()),
        "grains_under_5_pixels": int((pixels < 5).sum()),
        "grains_under_10_pixels": int((pixels < 10).sum()),
        "median_equivalent_diameter": float(diameter.median()),
        "mean_equivalent_diameter": float(diameter.mean()),
        "p90_equivalent_diameter": float(diameter.quantile(.90)),
        "maximum_equivalent_diameter": float(diameter.max())
    })
result = pd.DataFrame(rows)
csv_path = OUT / "ctf_segmentation_threshold_feature_sensitivity.csv"
json_path = OUT / "ctf_segmentation_threshold_feature_sensitivity.json"
result.to_csv(csv_path, index=False)
report = {
    "thresholds_compared": thresholds,
    "expected_indexed_pixels": expected_pixels,
    "all_pixel_conservation_checks_pass": bool(
        (result["indexed_pixels"] == expected_pixels).all()
    ),
    "results": result.replace(
        {np.nan: None}
    ).to_dict(orient="records"),
    "interpretation": [
        "All thresholds use the same indexed-pixel population.",
        "Higher thresholds merge more connected components.",
        "Grain-size statistics are segmentation-dependent.",
        "No threshold is declared experimentally validated by this comparison.",
        "CTF coordinate units remain physically unconfirmed."
    ]
}
json_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("=" * 72)
print("SEGMENTATION THRESHOLD FEATURE SENSITIVITY")
print("=" * 72)
print(result.to_string(index=False))
print("\nSaved:")
print(csv_path)
print(json_path)
