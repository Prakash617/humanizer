"""FastAPI application entry point.

Phase 1 contains only a small health-check-style scaffold. The real endpoints
and model lifecycle will be added in Phase 4, after inference is working in
Colab.
"""

from fastapi import FastAPI

from app import __version__

app = FastAPI(
    title="AI Humanizer API",
    version=__version__,
    description="A small API for improving the naturalness of English text.",
)


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Return basic API information so the local setup can be verified."""

    return {
        "name": "AI Humanizer API",
        "version": __version__,
        "status": "scaffold",
    }
