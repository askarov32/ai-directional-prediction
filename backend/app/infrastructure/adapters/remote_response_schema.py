from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.exceptions import MalformedRemoteResponseError


class RemotePredictionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    direction_vector: list[float]
    azimuth_deg: float = Field(..., allow_inf_nan=False)
    elevation_deg: float = Field(..., allow_inf_nan=False)
    magnitude: float = Field(..., ge=0, allow_inf_nan=False)
    wave_type: str = Field(..., min_length=1)
    travel_time_ms: float = Field(..., ge=0, allow_inf_nan=False)

    @field_validator("direction_vector")
    @classmethod
    def validate_direction_vector(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("direction_vector must contain exactly three values")
        if not all(math.isfinite(component) for component in value):
            raise ValueError("direction_vector must contain only finite values")
        if math.sqrt(sum(component * component for component in value)) == 0:
            raise ValueError("direction_vector magnitude must be greater than zero")
        return value


class RemoteFieldSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_displacement: float = Field(..., ge=0, allow_inf_nan=False)
    max_temperature_perturbation: float = Field(..., ge=0, allow_inf_nan=False)


class RemoteModelResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prediction: RemotePredictionPayload
    field_summary: RemoteFieldSummary
    model_version: str = Field(..., min_length=1)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], service_name: str) -> "RemoteModelResponse":
        candidate = _coerce_payload_shape(payload)
        try:
            return cls.model_validate(candidate)
        except ValidationError as exc:
            raise MalformedRemoteResponseError(
                service_name,
                {
                    "reason": "remote model response did not match the required schema",
                    "errors": exc.errors(),
                },
            ) from exc


def _coerce_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    model_version = payload.get("model_version")
    meta = payload.get("meta")
    if isinstance(meta, dict) and model_version is None:
        model_version = meta.get("model_version")

    if "prediction" in payload or "field_summary" in payload:
        return {
            "prediction": payload.get("prediction"),
            "field_summary": payload.get("field_summary"),
            "model_version": model_version,
        }

    return {
        "prediction": payload,
        "field_summary": payload,
        "model_version": model_version,
    }
