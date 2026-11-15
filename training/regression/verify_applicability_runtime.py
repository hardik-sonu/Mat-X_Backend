import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.engines.applicability_engine import assess_applicability
MODEL_DIR = ROOT / "models"
TARGETS = {
    "tensile_strength": "applicability_tensile_reference_v1.json",
    "yield_strength": "applicability_yield_reference_v1.json",
}
for target, filename in TARGETS.items():
    print("\n" + "=" * 65)
    print("TARGET:", target.upper())
    artifact = json.loads(
        (MODEL_DIR / filename).read_text(encoding="utf-8")
    )
    numeric_reference = artifact["reference_statistics"]["numeric"]
    categorical_reference = artifact["reference_statistics"]["categorical"]
    payload = {
        feature: numeric_reference[feature]["median"]
        for feature in artifact["features"]["numeric"]
    }
    for feature in artifact["features"]["categorical"]:
        payload[feature] = sorted(categorical_reference[feature])[0]
    # 1. Complete, known-category input
    normal = assess_applicability(target, payload)
    print("\nNORMAL INPUT:")
    print("Status:", normal["status"])
    print("Distance:", normal["distance"])
    print("Provided:", normal["provided_feature_count"])
    assert normal["distance"] is not None
    assert normal["provided_feature_count"] == 13
    # 2. Missing input
    incomplete_payload = payload.copy()
    incomplete_payload.pop("C")
    incomplete = assess_applicability(target, incomplete_payload)
    print("\nMISSING INPUT:")
    print("Status:", incomplete["status"])
    print("Missing:", incomplete["missing_features"])
    assert incomplete["status"] == "INSUFFICIENT_EVIDENCE"
    assert "C" in incomplete["missing_features"]
    assert incomplete["distance"] is None
    # 3. Unseen categorical value
    unseen_payload = payload.copy()
    unseen_payload["quench_medium"] = "MATX_UNSEEN_CATEGORY"
    unseen = assess_applicability(target, unseen_payload)
    print("\nUNSEEN CATEGORY:")
    print("Status:", unseen["status"])
    print("Unseen:", unseen["unseen_categories"])
    assert "quench_medium" in unseen["unseen_categories"]
    assert unseen["status"] in {"BORDERLINE", "OUT_OF_DOMAIN"}
    # 4. Extreme numeric value
    extreme_payload = payload.copy()
    extreme_payload["C"] = 1_000_000.0
    extreme = assess_applicability(target, extreme_payload)
    print("\nEXTREME NUMERIC INPUT:")
    print("Status:", extreme["status"])
    print("Distance:", extreme["distance"])
    print("Range violations:", extreme["range_violations"])
    assert extreme["status"] == "OUT_OF_DOMAIN"
    assert extreme["distance"] is not None
    print("\nTARGET VERIFICATION: PASS")
print("\nALL APPLICABILITY RUNTIME CHECKS PASSED.")
