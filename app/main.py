"""FastAPI application and HTTP routes.

The API is deliberately thin: it validates requests, calls the configured model
service, and translates domain errors into HTTP responses. Local inference lives in
``app.humanizer`` and remote inference lives in ``app.remote``.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.humanizer import (
    HumanizerError,
    HumanizerProtocol,
    InferenceError,
    InputTooLongError,
    InvalidInputError,
    InvalidStyleError,
    ModelNotLoadedError,
    get_humanizer,
)
from app.remote import (
    RemoteAuthError,
    RemoteConnectionError,
    RemoteHumanizer,
    RemoteRateLimitError,
    RemoteResponseError,
    RemoteTimeoutError,
)
from app.schemas import (
    HealthResponse,
    HumanizeRequest,
    HumanizeResponse,
    RootResponse,
)
from app.ui import UI_HTML

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_service() -> HumanizerProtocol:
    """Choose the configured inference backend."""

    if settings.inference_backend == "remote":
        return RemoteHumanizer()
    return get_humanizer()


def create_app(humanizer: HumanizerProtocol | None = None) -> FastAPI:
    """Build the API.

    ``humanizer`` can be injected in tests or in a Colab notebook that has already
    loaded the model. The default is the configured local or remote backend.
    """

    service = humanizer if humanizer is not None else _build_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        try:
            # Local model loading can block, so keep it off the event loop.
            await run_in_threadpool(service.load)
        except Exception:
            logger.exception("Backend failed to load during application startup")
            app.state.startup_error = "Model failed to load. Check the server logs."
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Rewrite English text for clarity, fluency, and readability while "
            "preserving meaning. The configured local or remote model handles "
            "the rewrite."
        ),
        lifespan=lifespan,
    )
    app.state.humanizer = service
    app.state.startup_error = None

    def require_api_key(
        x_api_key: Annotated[
            str | None,
            Header(alias="X-API-Key", description="Required when API_KEY is set."),
        ] = None,
    ) -> None:
        expected = settings.api_key
        if expected and (
            x_api_key is None or not secrets.compare_digest(x_api_key, expected)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "X-API-Key"},
            )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        for error in errors:
            location = error.get("loc", ())
            if error.get("type") == "string_too_long" and "text" in location:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "detail": (
                            "Input text is too long. Maximum "
                            f"{settings.max_input_characters} characters."
                        )
                    },
                )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(errors)},
        )

    @app.get("/", response_model=RootResponse, tags=["system"])
    def read_root() -> RootResponse:
        """Return the API name, version, status, and model readiness."""

        return RootResponse(
            name=settings.app_name,
            version=settings.app_version,
            status="running",
            model_loaded=service.is_loaded,
            docs_url="/docs",
            ui_url="/ui",
        )

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def humanizer_ui() -> HTMLResponse:
        """Serve the dependency-free browser UI."""

        return HTMLResponse(UI_HTML)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        responses={503: {"description": "Model is not loaded"}},
    )
    def health():
        """Report whether the process and model are ready."""

        loaded = service.is_loaded
        payload = HealthResponse(
            status="ok" if loaded else "degraded",
            version=settings.app_version,
            model_loaded=loaded,
            model_id=service.loaded_model_id if loaded else None,
            detail=(
                None
                if loaded
                else "Model is not loaded. Check the server logs and retry."
            ),
        )
        if not loaded:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )
        return payload

    @app.post(
        "/humanize",
        response_model=HumanizeResponse,
        tags=["humanization"],
        dependencies=[Depends(require_api_key)],
        responses={
            400: {"description": "Invalid input"},
            401: {"description": "Invalid or missing API key"},
            413: {"description": "Input too long"},
            422: {"description": "Validation error"},
            429: {"description": "Remote provider rate limit reached"},
            500: {"description": "Model inference failed"},
            502: {"description": "Remote provider rejected the request"},
            503: {"description": "Model is not loaded"},
            504: {"description": "Remote provider timed out"},
        },
    )
    def humanize(payload: HumanizeRequest) -> HumanizeResponse:
        """Rewrite text and return only the model-generated rewrite."""

        if not service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded yet. Check GET /health.",
            )

        try:
            rewritten = service.humanize_text(
                payload.text,
                style=payload.style.value,
            )
        except InputTooLongError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except InvalidStyleError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except InvalidInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except ModelNotLoadedError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except RemoteAuthError as exc:
            logger.warning("Remote provider rejected the configured credentials")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except RemoteRateLimitError as exc:
            logger.warning("Remote provider rate limit reached")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        except RemoteTimeoutError as exc:
            logger.warning("Remote provider timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc
        except RemoteResponseError as exc:
            logger.warning("Remote provider returned an unusable response")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except RemoteConnectionError as exc:
            logger.warning("Could not reach the remote provider")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except InferenceError as exc:
            logger.exception("Inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except HumanizerError as exc:
            logger.exception("Humanization failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Humanization failed.",
            ) from exc

        return HumanizeResponse(
            original_text=payload.text,
            humanized_text=rewritten,
            style=payload.style,
            original_characters=len(payload.text),
            rewritten_characters=len(rewritten),
        )

    return app


app = create_app()
