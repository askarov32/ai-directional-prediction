from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from fno_service.api.schemas import PredictionPayload
from fno_service.inference.predictor import CheckpointNotReadyError, FNOInferenceService


router = APIRouter()


def get_service(request: Request) -> FNOInferenceService:
    return request.app.state.fno_service


@router.get("/health")
async def health(request: Request) -> dict:
    return get_service(request).health_payload()


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    payload = get_service(request).readiness_payload()
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)


@router.post("/predict")
async def predict(request: Request, payload: PredictionPayload) -> dict:
    try:
        return get_service(request).predict(payload)
    except CheckpointNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CHECKPOINT_NOT_READY",
                "message": str(exc),
            },
        ) from exc
