"""Pydantic request and response schemas for the API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings

settings = get_settings()


class WritingStyle(str, Enum):
    """Supported writing styles."""

    natural = "natural"
    academic = "academic"
    professional = "professional"
    simple = "simple"
    friendly = "friendly"


class HumanizeRequest(BaseModel):
    """Body for POST /humanize."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Technology plays an important role in modern education.",
                    "style": "natural",
                }
            ]
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_input_characters,
        description=(
            "English text to rewrite. Empty or whitespace-only text is rejected."
        ),
    )
    style: WritingStyle = Field(
        default=WritingStyle.natural,
        description="Desired writing style.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty or whitespace-only.")
        return value


class HumanizeResponse(BaseModel):
    """Successful response from POST /humanize."""

    original_text: str
    humanized_text: str
    style: WritingStyle
    original_characters: int = Field(ge=0)
    rewritten_characters: int = Field(ge=0)


class RootResponse(BaseModel):
    """Basic service information."""

    name: str
    version: str
    status: str
    model_loaded: bool
    docs_url: str
    ui_url: str


class HealthResponse(BaseModel):
    """Health and model-readiness information."""

    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model_id: str | None = None
    detail: str | None = None
