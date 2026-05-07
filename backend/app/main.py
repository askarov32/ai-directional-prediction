from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_app_settings, get_medium_catalog_service
from app.api.routes.health import router as health_router
from app.api.routes.media import router as media_router
from app.api.routes.models import router as models_router
from app.api.routes.predictions import router as predictions_router
from app.core.exceptions import AppError
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_app_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("app")
    media = get_medium_catalog_service().list_media()
    logger.info("Loaded geological media catalog with %s entries", len(media))
    yield


settings = get_app_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(media_router, prefix=settings.api_prefix)
app.include_router(models_router, prefix=settings.api_prefix)
app.include_router(predictions_router, prefix=settings.api_prefix)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    started_at = perf_counter()

    response = await call_next(request)
    latency_ms = int((perf_counter() - started_at) * 1000)
    response.headers["X-Request-ID"] = request_id

    logging.getLogger("app.request").info(
        "request completed request_id=%s method=%s path=%s status_code=%s latency_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    location = ".".join(str(item) for item in first_error.get("loc", []))
    message = str(first_error.get("msg", "Request validation failed")).replace("Value error, ", "")

    if location.endswith("model"):
        code = "UNKNOWN_MODEL"
        message = "Unsupported model requested"
    elif "coordinates" in message.lower():
        code = "INVALID_COORDINATES"
    elif "resolution" in location or "rect_2d" in message.lower() or "rect_3d" in message.lower():
        code = "INVALID_RESOLUTION"
    else:
        code = "VALIDATION_ERROR"

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": {"errors": errors},
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("app").exception("Unhandled application error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            }
        },
    )
