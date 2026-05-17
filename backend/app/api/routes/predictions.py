from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.dependencies import (
    get_predict_direction_use_case,
    get_predict_direction_v2_use_case,
)
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.domain.use_cases.predict_direction_v2 import PredictDirectionV2UseCase
from app.schemas.prediction import (
    PredictionRequestSchema,
    PredictionRequestV2Schema,
)

router = APIRouter(tags=["predictions"])
logger = logging.getLogger("app.predictions")


def _detect_schema_version(body: dict[str, Any]) -> str:
    """Read schema_version; treat missing or '1.0' as the v1 path."""
    raw = body.get("schema_version")
    if raw is None:
        return "1.0"
    return str(raw)


@router.post("/predictions", response_model=None)
async def predict_direction(
    request: Request,
    v1_use_case: PredictDirectionUseCase = Depends(get_predict_direction_use_case),
    v2_use_case: PredictDirectionV2UseCase = Depends(
        get_predict_direction_v2_use_case
    ),
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", "unknown")
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_json", "message": str(exc)},
        ) from exc

    if not isinstance(body, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_payload",
                "message": "Request body must be a JSON object.",
            },
        )

    schema_version = _detect_schema_version(body)

    if schema_version == "2.0":
        try:
            payload_v2 = PredictionRequestV2Schema.model_validate(body)
        except ValidationError as exc:
            # Re-raise as a FastAPI request validation error so the
            # existing global handler builds the envelope.
            raise RequestValidationError(errors=exc.errors()) from exc
        logger.info(
            "prediction (v2) accepted request_id=%s model=%s medium_id=%s",
            request_id, payload_v2.model.value, payload_v2.medium_id,
        )
        result = await v2_use_case.execute(payload_v2.to_entity())
        logger.info(
            "prediction (v2) completed request_id=%s model=%s medium_id=%s",
            request_id, payload_v2.model.value, payload_v2.medium_id,
        )
        return result

    # v1 / legacy path — unchanged behaviour
    try:
        payload_v1 = PredictionRequestSchema.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError(errors=exc.errors()) from exc
    logger.info(
        "prediction (v1) accepted request_id=%s model=%s medium_id=%s",
        request_id, payload_v1.model.value, payload_v1.medium_id,
    )
    result_v1 = await v1_use_case.execute(payload_v1.to_entity())
    logger.info(
        "prediction (v1) completed request_id=%s model=%s medium_id=%s",
        request_id, payload_v1.model.value, payload_v1.medium_id,
    )
    return result_v1
