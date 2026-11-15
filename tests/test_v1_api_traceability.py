from app.main import app
def test_v1_material_operations_are_exposed_in_openapi():
    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/scientific-records/{table}"]
    assert "get" in paths["/api/v1/scientific-records/{table}"]
    assert "get" in paths[
        "/api/v1/scientific-records/{table}/{record_id}"
    ]
    assert "post" in paths[
        "/api/v1/material/state/reconstruct"
    ]
def test_v1_core_api_routes_are_exposed_in_openapi():
    paths = app.openapi()["paths"]
    required = {
        "/api/v1/health",
        "/api/v1/models",
        "/api/v1/microstructure/analyze",
        "/api/v1/investigation/recommend",
        "/api/v1/datasets/steelbench/records",
        "/api/v1/engineering/report",
    }
    assert required.issubset(paths)
