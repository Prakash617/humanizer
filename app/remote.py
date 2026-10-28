"""Remote inference through an OpenAI-compatible chat completions API.

This backend is intended for NVIDIA NIM, but it works with any provider that
implements the same request and response shape. It uses only the standard
library, so the lightweight API environment does not need PyTorch or
Transformers.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app import __version__
from app.config import get_settings
from app.humanizer import (
    STYLE_GUIDES,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    VALID_STYLES,
    InferenceError,
    InputTooLongError,
    InvalidInputError,
    InvalidStyleError,
    ModelLoadError,
    ModelNotLoadedError,
    clean_model_output,
)

logger = logging.getLogger(__name__)


class RemoteError(InferenceError):
    """Base class for remote-provider failures."""


class RemoteAuthError(RemoteError):
    """The provider rejected the configured credentials."""


class RemoteRateLimitError(RemoteError):
    """The provider asked the client to slow down."""


class RemoteTimeoutError(RemoteError):
    """The provider did not answer before the timeout."""


class RemoteConnectionError(RemoteError):
    """The provider could not be reached."""


class RemoteResponseError(RemoteError):
    """The provider returned an unusable response."""


def _content_to_text(content: object) -> str:
    """Extract text from either a string or an OpenAI content-part list."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class RemoteHumanizer:
    """Call a hosted OpenAI-compatible model for each rewrite request."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
        max_input_characters: int | None = None,
    ) -> None:
        settings = get_settings()

        raw_base_url = (
            settings.remote_base_url if base_url is None else base_url
        ).strip()
        if not raw_base_url:
            raise ValueError("remote base_url must not be empty.")

        self.base_url = raw_base_url.rstrip("/")
        self.endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self.model = (settings.remote_model if model is None else model).strip()
        self.api_key = (
            settings.remote_api_key if api_key is None else api_key
        )
        if self.api_key is not None:
            self.api_key = self.api_key.strip() or None

        self.timeout_seconds = (
            settings.remote_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_tokens = (
            settings.remote_max_tokens if max_tokens is None else max_tokens
        )
        self.temperature = (
            settings.temperature if temperature is None else temperature
        )
        self.top_p = settings.top_p if top_p is None else top_p
        self.extra_body = dict(
            settings.remote_extra_body if extra_body is None else extra_body
        )
        self.max_input_characters = (
            settings.max_input_characters
            if max_input_characters is None
            else max_input_characters
        )

        if not self.model:
            raise ValueError("remote model must not be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("remote timeout_seconds must be positive.")
        if self.max_tokens < 1:
            raise ValueError("remote max_tokens must be positive.")
        if self.temperature < 0:
            raise ValueError("remote temperature cannot be negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("remote top_p must be greater than 0 and at most 1.")
        if self.max_input_characters < 1:
            raise ValueError("remote max_input_characters must be positive.")

        self._loaded = False
        self._loaded_model_id: str | None = None
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self, *, force: bool = False) -> None:
        """Validate configuration without making a network request."""

        if self._loaded and not force:
            return

        if force:
            self._loaded = False
            self._loaded_model_id = None
            self._load_error = None

        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            message = (
                "REMOTE_BASE_URL must be an absolute http:// or https:// URL. "
                f"Got {self.base_url!r}."
            )
            self._load_error = message
            raise ModelLoadError(message)

        if not self.api_key:
            message = (
                "REMOTE_API_KEY is required when INFERENCE_BACKEND=remote. "
                "Add it to .env; do not commit it."
            )
            self._load_error = message
            raise ModelLoadError(message)

        self._loaded = True
        self._loaded_model_id = self.model
        self._load_error = None
        logger.info(
            "Remote backend ready: %s at %s",
            self.model,
            self.base_url,
        )

    def _require_loaded(self) -> None:
        if self._loaded:
            return

        try:
            self.load()
        except ModelLoadError as exc:
            raise ModelNotLoadedError(str(exc)) from exc

        if not self._loaded:
            raise ModelNotLoadedError("The remote backend is not loaded.")

    def _build_payload(
        self,
        text: str,
        style: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    style=style,
                    style_guide=STYLE_GUIDES[style],
                    text=text,
                ),
            },
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Provider-specific fields such as Nemotron's chat_template_kwargs belong
        # in REMOTE_EXTRA_BODY. Core request fields cannot be overridden by it.
        for key, value in self.extra_body.items():
            if key not in {"model", "messages", "stream"}:
                payload[key] = value

        return payload

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"ai-humanizer/{__version__}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            logger.warning(
                "Remote provider returned HTTP %s for %s",
                exc.code,
                self.base_url,
            )
            if exc.code in {401, 403}:
                raise RemoteAuthError(
                    "The remote provider rejected the API key."
                ) from exc
            if exc.code == 429:
                raise RemoteRateLimitError(
                    "The remote provider rate limit was reached. Retry later."
                ) from exc
            raise RemoteResponseError(
                f"The remote provider returned HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)) or (
                "timed out" in str(reason).lower()
            ):
                raise RemoteTimeoutError(
                    "The remote provider did not answer before the timeout."
                ) from exc
            logger.warning(
                "Could not reach remote provider %s: %s",
                self.base_url,
                type(reason).__name__,
            )
            raise RemoteConnectionError(
                "Could not reach the remote provider."
            ) from exc
        except TimeoutError as exc:
            raise RemoteTimeoutError(
                "The remote provider did not answer before the timeout."
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteResponseError(
                "The remote provider returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RemoteResponseError(
                "The remote provider returned an unexpected response."
            )
        return data

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RemoteResponseError(
                "The remote provider response did not contain choices."
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            raise RemoteResponseError(
                "The remote provider returned an unexpected choice."
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise RemoteResponseError(
                "The remote provider response did not contain a message."
            )

        rewritten = clean_model_output(_content_to_text(message.get("content")))
        if not rewritten:
            raise RemoteResponseError(
                "The remote provider returned an empty rewrite."
            )
        return rewritten

    def humanize_text(
        self,
        text: str,
        style: str = "natural",
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Rewrite text through the remote provider and return only the rewrite."""

        if not isinstance(text, str):
            raise InvalidInputError("text must be a string.")
        if not text.strip():
            raise InvalidInputError("Input text must not be empty.")

        style_key = style.strip().lower() if isinstance(style, str) else ""
        if style_key not in VALID_STYLES:
            raise InvalidStyleError(
                f"Unknown style {style!r}. Choose from: {', '.join(VALID_STYLES)}."
            )

        if len(text) > self.max_input_characters:
            raise InputTooLongError(
                f"Input is {len(text)} characters, above the "
                f"{self.max_input_characters}-character safe limit."
            )

        self._require_loaded()

        effective_temperature = (
            self.temperature if temperature is None else temperature
        )
        effective_top_p = self.top_p if top_p is None else top_p
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if effective_temperature < 0:
            raise InvalidInputError("temperature cannot be negative.")
        if not 0 < effective_top_p <= 1:
            raise InvalidInputError("top_p must be greater than 0 and at most 1.")
        if effective_max_tokens < 1:
            raise InvalidInputError("max_tokens must be positive.")

        payload = self._build_payload(
            text,
            style_key,
            temperature=effective_temperature,
            top_p=effective_top_p,
            max_tokens=effective_max_tokens,
        )
        data = self._request(payload)
        return self._extract_text(data)
