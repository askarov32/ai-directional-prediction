from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MediumPropertiesSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rho: float = Field(..., gt=0, allow_inf_nan=False)
    porosity_total: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    porosity_effective: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    vp: float = Field(..., gt=0, allow_inf_nan=False)
    vs: float = Field(..., gt=0, allow_inf_nan=False)
    thermal_conductivity: float = Field(..., gt=0, allow_inf_nan=False)
    heat_capacity: float = Field(..., gt=0, allow_inf_nan=False)
    thermal_expansion: float = Field(..., ge=0, allow_inf_nan=False)


class MediumSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    category: str
    properties: MediumPropertiesSchema
    ranges: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ScenarioSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_c: float = Field(..., ge=-273.15, le=2000, allow_inf_nan=False)
    pressure_mpa: float = Field(..., gt=0, le=5000, allow_inf_nan=False)
    time_ms: float = Field(..., ge=0, le=60_000, allow_inf_nan=False)


class SourceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    x: float = Field(..., ge=0, allow_inf_nan=False)
    y: float = Field(..., ge=0, allow_inf_nan=False)
    z: float = Field(..., ge=0, allow_inf_nan=False)
    amplitude: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    frequency_hz: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    direction: list[float]

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("Source direction must have exactly 3 components.")
        if not all(math.isfinite(component) for component in value):
            raise ValueError("Source direction components must be finite.")
        if math.sqrt(sum(component * component for component in value)) == 0:
            raise ValueError("Source direction magnitude must be greater than zero.")
        return value


class ProbeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0, allow_inf_nan=False)
    y: float = Field(..., ge=0, allow_inf_nan=False)
    z: float = Field(..., ge=0, allow_inf_nan=False)


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


class DomainSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rect_2d", "rect_3d"]
    size: DomainSizeSchema
    resolution: DomainResolutionSchema
    boundary_conditions: BoundaryConditionsSchema


class PINNPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medium: MediumSchema
    scenario: ScenarioSchema
    source: SourceSchema
    probe: ProbeSchema
    domain: DomainSchema
    representation: Literal["physics_informed"]
    routing_hint: Literal["pinn"]
