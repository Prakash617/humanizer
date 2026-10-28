"""Application configuration loaded from environment variables or .env.

Local inference needs PyTorch and Transformers. Remote inference only needs an
OpenAI-compatible endpoint and a provider key.
"""

import json
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app import __version__


class Settings(BaseSettings):
    """Runtime settings.

    Pydantic Settings reads matching environment variables and then `.env`.
    Environment variables take precedence over the file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AI Humanizer API"
    app_version: str = __version__
    log_level: str = "INFO"

    # "local" uses the Hugging Face model in app/humanizer.py. "remote" uses an
    # OpenAI-compatible endpoint, defaulting to NVIDIA NIM.
    inference_backend: Literal["local", "remote"] = "local"

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_token: str | None = None

    remote_base_url: str = "https://integrate.api.nvidia.com/v1"
    remote_model: str = "nvidia/nemotron-3-super-120b-a12b"
    remote_api_key: str | None = None
    remote_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    remote_max_tokens: int = Field(default=1024, ge=32, le=32768)
    # Nemotron 3 Super reasons by default. Thinking adds latency and is unnecessary
    # for editing. Set REMOTE_EXTRA_BODY= to an empty value to disable this.
    remote_extra_body: Annotated[dict[str, Any], NoDecode] = Field(
        default_factory=lambda: {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )

    max_input_characters: int = Field(default=8000, ge=1)
    max_input_tokens: int = Field(default=6000, ge=250)
    max_chunk_tokens: int = Field(default=1400, ge=128)
    max_new_tokens: int = Field(default=1024, ge=32)

    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    repetition_penalty: float = Field(default=1.1, gt=0.0)

    # Optional. When set, POST /humanize requires the X-API-Key header.
    api_key: str | None = None

    @field_validator("inference_backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("remote_base_url", mode="before")
    @classmethod
    def normalize_remote_base_url(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().rstrip("/")
            if not value:
                raise ValueError("REMOTE_BASE_URL must not be empty.")
        return value

    @field_validator("remote_model", mode="before")
    @classmethod
    def normalize_remote_model(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("REMOTE_MODEL must not be empty.")
        return value

    @field_validator("remote_extra_body", mode="before")
    @classmethod
    def parse_extra_body(cls, value: object) -> object:
        if value is None:
            return {}
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "REMOTE_EXTRA_BODY must be a valid JSON object."
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError("REMOTE_EXTRA_BODY must be a JSON object.")
            return parsed
        return value

    @field_validator("hf_token", "api_key", "remote_api_key", mode="before")
    @classmethod
    def blank_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one settings object per process."""

    return Settings()
