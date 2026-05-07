from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_predict_direction_use_case
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.schemas.prediction import PredictionRequestSchema, PredictionResponseSchema

router = APIRouter(tags=["predictions"])
logger = logging.getLogger("app.predictions")


@router.post("/predictions", response_model=PredictionResponseSchema)
async def predict_direction(
    request: Request,
    payload: PredictionRequestSchema,
    use_case: PredictDirectionUseCase = Depends(get_predict_direction_use_case),
) -> PredictionResponseSchema:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        "prediction request accepted request_id=%s model=%s medium_id=%s",
        request_id,
        payload.model.value,
        payload.medium_id,
    )
    result = await use_case.execute(payload.to_entity())
    logger.info(
        "prediction request completed request_id=%s model=%s medium_id=%s",
        request_id,
        payload.model.value,
        payload.medium_id,
    )
    return PredictionResponseSchema.model_validate(result)
