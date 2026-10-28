"""Real-model integration test.

Run this only in an environment with PyTorch, Transformers, and enough memory:

    RUN_REAL_MODEL_TESTS=1 python -m pytest tests/test_integration_real_model.py -q -s

It is skipped by default so ordinary local tests never download a model.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.humanizer import ModelHumanizer
from app.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS") != "1",
        reason="Set RUN_REAL_MODEL_TESTS=1 to run the real model integration test.",
    ),
]


def test_real_model_loads_and_humanizes() -> None:
    service = ModelHumanizer()
    app = create_app(service)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["model_loaded"] is True

        response = client.post(
            "/humanize",
            json={
                "text": "Technology plays an important role in modern education.",
                "style": "natural",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["original_text"]
    assert body["humanized_text"].strip()
    assert body["style"] == "natural"
    assert body["original_characters"] == len(body["original_text"])
    assert body["rewritten_characters"] == len(body["humanized_text"])
