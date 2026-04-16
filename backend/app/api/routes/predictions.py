from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_predict_direction_use_case
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.schemas.prediction import PredictionRequestSchema, PredictionResponseSchema

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionResponseSchema)
async def predict_direction(
    payload: PredictionRequestSchema,
    use_case: PredictDirectionUseCase = Depends(get_predict_direction_use_case),
) -> PredictionResponseSchema:
    result = await use_case.execute(payload.to_entity())
    return PredictionResponseSchema.model_validate(result)
