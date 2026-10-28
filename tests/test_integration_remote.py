"""Opt-in integration test for the remote OpenAI-compatible backend.

This test makes a real network request and may incur provider charges:

    RUN_REMOTE_TESTS=1 python -m pytest tests/test_integration_remote.py -q -s

It reads REMOTE_API_KEY from the environment or .env. The key is never printed.
"""

from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.remote import RemoteHumanizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_REMOTE_TESTS") != "1",
        reason="Set RUN_REMOTE_TESTS=1 to call the remote provider.",
    ),
]


def test_remote_provider_rewrites_and_preserves_details() -> None:
    settings = get_settings()
    if not settings.remote_api_key:
        pytest.skip("Set REMOTE_API_KEY in .env before running this test.")

    service = RemoteHumanizer()
    service.load()

    text = (
        "In 2024, Dr. Chen published 3 reports about remote work. "
        "The reports covered 120 employees in Berlin."
    )
    rewritten = service.humanize_text(text, style="professional")

    assert rewritten.strip()
    assert rewritten != text
    assert "2024" in rewritten
    assert "3" in rewritten
    assert "120" in rewritten
    assert "Chen" in rewritten
    assert "Berlin" in rewritten
