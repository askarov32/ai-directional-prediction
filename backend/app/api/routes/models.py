from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_prediction_router
from app.domain.services.prediction_router import PredictionRouter
from app.schemas.prediction import ModelInfoSchema

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelInfoSchema])
async def list_models(
    prediction_router: PredictionRouter = Depends(get_prediction_router),
) -> list[ModelInfoSchema]:
    return [ModelInfoSchema.model_validate(item) for item in prediction_router.list_models()]
