"""Offline tests for the OpenAI-compatible remote backend.

No test in this file makes a network request. ``urlopen`` is replaced with a
small fake so request construction and error mapping can be checked offline.
"""

from __future__ import annotations

import io
import json
import socket
from urllib.error import HTTPError, URLError

import pytest
from fastapi.testclient import TestClient

import app.remote as remote_module
from app.config import Settings
from app.humanizer import (
    InputTooLongError,
    InvalidInputError,
    InvalidStyleError,
    ModelLoadError,
)
from app.main import create_app, settings
from app.remote import (
    RemoteAuthError,
    RemoteConnectionError,
    RemoteHumanizer,
    RemoteRateLimitError,
    RemoteResponseError,
    RemoteTimeoutError,
)


class FakeResponse:
    """Minimal context-managed response returned by the patched urlopen."""

    def __init__(self, payload: dict | bytes) -> None:
        if isinstance(payload, bytes):
            self._body = payload
        else:
            self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def make_remote(**overrides: object) -> RemoteHumanizer:
    defaults: dict[str, object] = {
        "base_url": "https://nim.example/v1",
        "model": "nvidia/test-model",
        "api_key": "test-key",
        "timeout_seconds": 5.0,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_input_characters": 8000,
    }
    defaults.update(overrides)
    return RemoteHumanizer(**defaults)


def install_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict | bytes,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
    return captured


def install_http_error(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code: int,
) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            code,
            "provider error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"provider detail"}'),
        )

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)


def test_remote_success_sends_expected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    remote.load()
    captured = install_response(
        monkeypatch,
        {"choices": [{"message": {"content": "  A clearer version.  "}}]},
    )

    result = remote.humanize_text(
        "In 2024, Dr. Chen wrote 3 reports.",
        style="academic",
    )

    assert result == "A clearer version."
    assert captured["url"] == "https://nim.example/v1/chat/completions"
    assert captured["timeout"] == 5.0

    headers = captured["headers"]
    assert headers["authorization"] == "Bearer test-key"
    assert headers["content-type"] == "application/json"

    payload = captured["payload"]
    assert payload["model"] == "nvidia/test-model"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["role"] == "system"
    assert "academic" in payload["messages"][1]["content"]
    assert "In 2024, Dr. Chen wrote 3 reports." in payload["messages"][1]["content"]


def test_remote_accepts_content_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = make_remote()
    remote.load()
    install_response(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello "},
                            {"type": "text", "text": "world."},
                        ]
                    }
                }
            ]
        },
    )

    assert remote.humanize_text("Hello world.") == "Hello world."


def test_remote_missing_key_fails_load() -> None:
    remote = make_remote(api_key="")

    with pytest.raises(ModelLoadError, match="REMOTE_API_KEY"):
        remote.load()

    assert remote.is_loaded is False
    assert remote.load_error


def test_remote_invalid_base_url_fails_load() -> None:
    remote = make_remote(base_url="not-a-url")

    with pytest.raises(ModelLoadError, match="REMOTE_BASE_URL"):
        remote.load()

    assert remote.is_loaded is False


def test_remote_rejects_bad_input_before_network() -> None:
    remote = make_remote(max_input_characters=10)

    with pytest.raises(InvalidInputError):
        remote.humanize_text("   ")
    with pytest.raises(InvalidStyleError):
        remote.humanize_text("Short text.", style="poetic")
    with pytest.raises(InputTooLongError):
        remote.humanize_text("x" * 11)


def test_remote_auth_failure_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = make_remote()
    remote.load()
    install_http_error(monkeypatch, code=401)

    with pytest.raises(RemoteAuthError) as error:
        remote.humanize_text("A short sentence.")

    assert "test-key" not in str(error.value)


def test_remote_rate_limit_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = make_remote()
    remote.load()
    install_http_error(monkeypatch, code=429)

    with pytest.raises(RemoteRateLimitError):
        remote.humanize_text("A short sentence.")


def test_remote_timeout_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = make_remote()
    remote.load()

    def fake_urlopen(request, timeout):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)

    with pytest.raises(RemoteTimeoutError):
        remote.humanize_text("A short sentence.")


def test_remote_connection_error_is_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    remote.load()

    def fake_urlopen(request, timeout):
        raise URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)

    with pytest.raises(RemoteConnectionError):
        remote.humanize_text("A short sentence.")


def test_remote_invalid_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    remote.load()
    install_response(monkeypatch, b"not json")

    with pytest.raises(RemoteResponseError, match="invalid JSON"):
        remote.humanize_text("A short sentence.")


def test_remote_missing_choices_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    remote.load()
    install_response(monkeypatch, {"choices": []})

    with pytest.raises(RemoteResponseError, match="choices"):
        remote.humanize_text("A short sentence.")


def test_remote_empty_content_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    remote.load()
    install_response(monkeypatch, {"choices": [{"message": {"content": "   "}}]})

    with pytest.raises(RemoteResponseError, match="empty"):
        remote.humanize_text("A short sentence.")


def test_api_uses_remote_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "inference_backend", "remote")
    monkeypatch.setattr(settings, "remote_api_key", "test-key")
    monkeypatch.setattr(settings, "remote_model", "nvidia/test-model")

    with TestClient(create_app()) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["model_id"] == "nvidia/test-model"
    assert health.json()["backend"] == "remote"
    assert health.json()["max_input_characters"] == settings.max_input_characters


def test_api_remote_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    install_response(
        monkeypatch,
        {"choices": [{"message": {"content": "A clearer version."}}]},
    )

    with TestClient(create_app(remote)) as client:
        response = client.post(
            "/humanize",
            json={"text": "In 2024, Dr. Chen wrote 3 reports.", "style": "natural"},
        )

    assert response.status_code == 200
    assert response.json()["humanized_text"] == "A clearer version."


def test_api_remote_auth_failure_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()
    with TestClient(create_app(remote)) as client:
        install_http_error(monkeypatch, code=401)
        response = client.post(
            "/humanize",
            json={"text": "A short sentence."},
        )

    assert response.status_code == 502
    assert "API key" in response.json()["detail"]
    assert "test-key" not in response.text


def test_api_remote_timeout_returns_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = make_remote()

    def fake_urlopen(request, timeout):
        raise URLError(socket.timeout("timed out"))

    with TestClient(create_app(remote)) as client:
        monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
        response = client.post(
            "/humanize",
            json={"text": "A short sentence."},
        )

    assert response.status_code == 504


def test_api_remote_missing_key_is_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "inference_backend", "remote")
    monkeypatch.setattr(settings, "remote_api_key", None)

    with TestClient(create_app()) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["model_loaded"] is False


def test_remote_settings_parse_json_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "REMOTE_EXTRA_BODY",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
    )

    parsed = Settings(_env_file=None)

    assert parsed.remote_extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_remote_settings_allow_blank_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REMOTE_EXTRA_BODY", "")

    parsed = Settings(_env_file=None)

    assert parsed.remote_extra_body == {}


def test_remote_backend_name_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "  REMOTE  ")
    monkeypatch.setenv("REMOTE_API_KEY", "   ")

    parsed = Settings(_env_file=None)

    assert parsed.inference_backend == "remote"
    assert parsed.remote_api_key is None
