"""Phase 1 smoke tests for the API scaffold.

The complete endpoint and model-mocking tests will be added in Phase 5.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_scaffold_metadata() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "AI Humanizer API",
        "version": "0.1.0",
        "status": "scaffold",
    }
