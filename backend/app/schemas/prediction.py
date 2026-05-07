from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.entities.prediction import (
    BoundaryConditions,
    Domain,
    DomainResolution,
    DomainSize,
    Probe,
    Scenario,
    Source,
    UnifiedPredictionRequest,
)
from app.domain.enums.model_type import ModelType


def _normalize_vector(direction: list[float]) -> tuple[float, float, float]:
    if any(not math.isfinite(component) for component in direction):
        raise ValueError("Direction vector components must be finite.")
    magnitude = math.sqrt(sum(component * component for component in direction))
    if magnitude == 0:
        raise ValueError("Direction vector magnitude must be greater than zero.")
    return tuple(component / magnitude for component in direction)


class ScenarioRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_c: float = Field(..., ge=-273.15, le=2000, allow_inf_nan=False)
    pressure_mpa: float = Field(..., gt=0, le=5000, allow_inf_nan=False)
    time_ms: float = Field(..., gt=0, le=60_000, allow_inf_nan=False)


class SourceRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., min_length=2, max_length=64)
    x: float = Field(..., ge=0, allow_inf_nan=False)
    y: float = Field(..., ge=0, allow_inf_nan=False)
    z: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    amplitude: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    frequency_hz: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    direction: list[float]

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("Direction vector must contain exactly three values.")
        _normalize_vector([float(item) for item in value])
        return value


class ProbeRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0, allow_inf_nan=False)
    y: float = Field(..., ge=0, allow_inf_nan=False)
    z: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class DomainSizeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lx: float = Field(..., gt=0, le=10_000, allow_inf_nan=False)
    ly: float = Field(..., gt=0, le=10_000, allow_inf_nan=False)
    lz: float = Field(..., ge=0, le=10_000, allow_inf_nan=False)


class DomainResolutionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nx: int = Field(..., ge=2, le=2048)
    ny: int = Field(..., ge=2, le=2048)
    nz: int = Field(..., ge=1, le=512)


class BoundaryConditionsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    top: str
    bottom: str
    front: str | None = None
    back: str | None = None


class DomainRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rect_2d", "rect_3d"]
    size: DomainSizeSchema
    resolution: DomainResolutionSchema
    boundary_conditions: BoundaryConditionsSchema

    @model_validator(mode="after")
    def validate_dimension_consistency(self) -> "DomainRequestSchema":
        if self.type == "rect_2d" and self.resolution.nz != 1:
            raise ValueError("For rect_2d domains, resolution.nz must be 1.")
        if self.type == "rect_2d" and self.size.lz != 0:
            raise ValueError("For rect_2d domains, size.lz must be 0.")
        if self.type == "rect_3d" and self.size.lz <= 0:
            raise ValueError("For rect_3d domains, size.lz must be greater than 0.")
        return self


class PredictionRequestSchema(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    model: ModelType
    medium_id: str = Field(..., min_length=2, max_length=120)
    scenario: ScenarioRequestSchema
    source: SourceRequestSchema
    probe: ProbeRequestSchema
    domain: DomainRequestSchema

    @model_validator(mode="after")
    def validate_coordinates(self) -> "PredictionRequestSchema":
        self._validate_point("source", self.source.x, self.source.y, self.source.z)
        self._validate_point("probe", self.probe.x, self.probe.y, self.probe.z)
        return self

    def _validate_point(self, label: str, x: float, y: float, z: float) -> None:
        size = self.domain.size
        if not (0 <= x <= size.lx and 0 <= y <= size.ly and 0 <= z <= size.lz):
            raise ValueError(
                f"Invalid coordinates for {label}. Coordinates must lie within the configured domain bounds."
            )

    def to_entity(self) -> UnifiedPredictionRequest:
        return UnifiedPredictionRequest(
            model=self.model,
            medium_id=self.medium_id,
            scenario=Scenario(
                temperature_c=self.scenario.temperature_c,
                pressure_mpa=self.scenario.pressure_mpa,
                time_ms=self.scenario.time_ms,
            ),
            source=Source(
                type=self.source.type,
                x=self.source.x,
                y=self.source.y,
                z=self.source.z,
                amplitude=self.source.amplitude,
                frequency_hz=self.source.frequency_hz,
                direction=_normalize_vector(self.source.direction),
            ),
            probe=Probe(x=self.probe.x, y=self.probe.y, z=self.probe.z),
            domain=Domain(
                type=self.domain.type,
                size=DomainSize(
                    lx=self.domain.size.lx,
                    ly=self.domain.size.ly,
                    lz=self.domain.size.lz,
                ),
                resolution=DomainResolution(
                    nx=self.domain.resolution.nx,
                    ny=self.domain.resolution.ny,
                    nz=self.domain.resolution.nz,
                ),
                boundary_conditions=BoundaryConditions(
                    left=self.domain.boundary_conditions.left,
                    right=self.domain.boundary_conditions.right,
                    top=self.domain.boundary_conditions.top,
                    bottom=self.domain.boundary_conditions.bottom,
                    front=self.domain.boundary_conditions.front,
                    back=self.domain.boundary_conditions.back,
                ),
            ),
        )


class MediumSummarySchema(BaseModel):
    id: str
    name: str
    category: str


class PredictionSummarySchema(BaseModel):
    direction_vector: list[float]
    azimuth_deg: float
    elevation_deg: float
    magnitude: float
    wave_type: str
    travel_time_ms: float


class FieldSummarySchema(BaseModel):
    max_displacement: float
    max_temperature_perturbation: float


class PredictionMetaSchema(BaseModel):
    model_version: str
    latency_ms: int
    request_id: str


class PredictionResponseSchema(BaseModel):
    model: ModelType
    medium: MediumSummarySchema
    prediction: PredictionSummarySchema
    field_summary: FieldSummarySchema
    meta: PredictionMetaSchema


class ModelInfoSchema(BaseModel):
    id: str
    name: str
    status: str
