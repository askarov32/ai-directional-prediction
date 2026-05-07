from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_medium_catalog_service, get_prediction_router
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.services.prediction_router import PredictionRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "thermoelastic-direction-api"}


@router.get("/ready")
async def ready(
    medium_catalog: MediumCatalogService = Depends(get_medium_catalog_service),
    prediction_router: PredictionRouter = Depends(get_prediction_router),
) -> JSONResponse:
    media = medium_catalog.list_media()
    model_checks = await prediction_router.readiness()
    catalog_ready = len(media) > 0
    models_ready = all(item.get("ready") is True for item in model_checks)
    ready_status = catalog_ready and models_ready

    return JSONResponse(
        status_code=status.HTTP_200_OK if ready_status else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ready_status else "not_ready",
            "service": "thermoelastic-direction-api",
            "checks": {
                "media_catalog": {
                    "ready": catalog_ready,
                    "count": len(media),
                },
                "models": model_checks,
            },
        },
    )
