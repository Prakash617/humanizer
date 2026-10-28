"""Application configuration loaded from environment variables or .env.

The laptop and Colab share the same settings names, but model inference is only
performed where PyTorch and Transformers are installed.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_token: str | None = None

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

    @field_validator("hf_token", "api_key", mode="before")
    @classmethod
    def blank_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one settings object per process."""

    return Settings()
