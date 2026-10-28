"""Unit tests for the FastAPI layer.

The model is always mocked here. These tests do not download weights, import
PyTorch, or require a GPU.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.humanizer import InferenceError, ModelLoadError
from app.main import create_app, settings
from app.schemas import HumanizeResponse, WritingStyle

settings = get_settings()


class FakeHumanizer:
    """Small stand-in for ModelHumanizer."""

    def __init__(
        self,
        *,
        load_ok: bool = True,
        inference_error: bool = False,
        output: str = "This is a clearer and more natural version.",
    ) -> None:
        self.load_ok = load_ok
        self.inference_error = inference_error
        self.output = output
        self._loaded = False
        self._loaded_model_id: str | None = None
        self._load_error: str | None = None
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if not self.load_ok:
            self._load_error = "simulated load failure"
            raise ModelLoadError("simulated load failure")
        self._loaded = True
        self._loaded_model_id = "fake/model"

    def humanize_text(self, text: str, style: str = "natural") -> str:
        if self.inference_error:
            raise InferenceError("simulated inference failure")
        return self.output

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    @property
    def load_error(self) -> str | None:
        return self._load_error


@pytest.fixture
def fake_humanizer() -> FakeHumanizer:
    return FakeHumanizer()


@pytest.fixture
def client(fake_humanizer: FakeHumanizer):
    with TestClient(create_app(fake_humanizer)) as test_client:
        yield test_client


def test_home_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == settings.app_name
    assert body["version"] == settings.app_version
    assert body["status"] == "running"
    assert body["model_loaded"] is True
    assert body["docs_url"] == "/docs"


def test_health_endpoint_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == "fake/model"
    assert body["detail"] is None


def test_health_endpoint_when_model_failed() -> None:
    fake = FakeHumanizer(load_ok=False)
    with TestClient(create_app(fake)) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_id"] is None


def test_valid_humanization_request(client: TestClient) -> None:
    text = "Technology plays an important role in modern education."
    response = client.post("/humanize", json={"text": text, "style": "natural"})

    assert response.status_code == 200
    body = response.json()
    assert body["original_text"] == text
    assert body["humanized_text"] == "This is a clearer and more natural version."
    assert body["style"] == "natural"
    assert body["original_characters"] == len(text)
    assert body["rewritten_characters"] == len(body["humanized_text"])


def test_empty_input_is_rejected(client: TestClient) -> None:
    response = client.post("/humanize", json={"text": ""})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_whitespace_only_input_is_rejected(client: TestClient) -> None:
    response = client.post("/humanize", json={"text": "   \n\t  "})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_missing_text_is_rejected(client: TestClient) -> None:
    response = client.post("/humanize", json={"style": "natural"})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_invalid_style_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/humanize",
        json={"text": "A short sentence.", "style": "poetic"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]


def test_excessively_long_input_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/humanize",
        json={"text": "x" * (settings.max_input_characters + 1)},
    )

    assert response.status_code == 413
    assert "too long" in response.json()["detail"].lower()


def test_model_inference_failure_returns_500() -> None:
    fake = FakeHumanizer(inference_error=True)
    with TestClient(create_app(fake)) as test_client:
        response = test_client.post(
            "/humanize",
            json={"text": "A short sentence."},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "simulated inference failure"


def test_humanize_when_model_not_loaded_returns_503() -> None:
    fake = FakeHumanizer(load_ok=False)
    with TestClient(create_app(fake)) as test_client:
        response = test_client.post(
            "/humanize",
            json={"text": "A short sentence."},
        )

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_response_matches_schema(client: TestClient) -> None:
    response = client.post(
        "/humanize",
        json={"text": "A short sentence.", "style": "friendly"},
    )

    assert response.status_code == 200
    parsed = HumanizeResponse.model_validate(response.json())
    assert parsed.style is WritingStyle.friendly
    assert parsed.original_characters == len(parsed.original_text)
    assert parsed.rewritten_characters == len(parsed.humanized_text)


def test_api_key_can_protect_humanize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", "test-secret")
    fake = FakeHumanizer()

    with TestClient(create_app(fake)) as test_client:
        missing = test_client.post("/humanize", json={"text": "A short sentence."})
        wrong = test_client.post(
            "/humanize",
            json={"text": "A short sentence."},
            headers={"X-API-Key": "wrong"},
        )
        correct = test_client.post(
            "/humanize",
            json={"text": "A short sentence."},
            headers={"X-API-Key": "test-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert correct.status_code == 200
