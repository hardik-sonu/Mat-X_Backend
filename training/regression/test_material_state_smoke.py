import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
endpoint = "/api/v1/material/state/reconstruct"
cases = [
    (
        "MINIMAL_EVIDENCE",
        {
            "material_id": "MATX-TEST-STATE-001",
            "material_family": "low_alloy_steel"
        }
    ),
    (
        "COMPOSITION_AND_MEASUREMENTS",
        {
            "material_id": "MATX-TEST-STATE-002",
            "material_family": "low_alloy_normalized",
            "composition": {
                "C": 0.30,
                "Mn": 0.80,
                "Cr": 1.0,
                "Mo": 0.20
            },
            "measured_properties": {
                "tensile_strength": 720,
                "yield_strength": 560
            }
        }
    ),
    (
        "UNSUPPORTED_REFERENCE_FAMILY",
        {
            "material_id": "MATX-TEST-STATE-003",
            "material_family": "low_alloy_steel",
            "composition": {
                "C": 0.30,
                "Mn": 0.80,
                "Cr": 1.0,
                "Mo": 0.20
            },
            "measured_properties": {
                "tensile_strength": 720,
                "yield_strength": 560
            }
        }
    ),
    (
        "INVALID_EMPTY_ID",
        {
            "material_id": "",
            "material_family": "low_alloy_steel"
        }
    ),
]
for name, payload in cases:
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)
    response = client.post(endpoint, json=payload)
    print("HTTP:", response.status_code)
    try:
        result = response.json()
    except Exception:
        print(response.text[:2000])
        continue
    if response.status_code == 200:
        print("Status:", result.get("status"))
        print("Material:", result.get("material"))
        print("Evidence summary:", result.get("evidence_summary"))
        print("Identity status:", (result.get("material_identity_assessment") or {}).get("status"))
        print("Investigation status:", (result.get("active_investigation") or {}).get("status"))
        print("Evidence categories:", list(result.get("evidence", {}).keys()))
        print("Scientific scope:", result.get("scientific_scope"))
    else:
        print("Detail:", result.get("detail"))
print("\n=== RECONSTRUCTION API SMOKE TEST COMPLETE ===")



