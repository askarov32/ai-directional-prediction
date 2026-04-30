from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MediumPropertiesSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    rho: float
    vp: float
    vs: float
    thermal_conductivity: float
    heat_capacity: float
    thermal_expansion: float


class MediumSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    category: str
    properties: MediumPropertiesSchema


class ScenarioSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    temperature_c: float
    pressure_mpa: float
    time_ms: float = Field(..., ge=0)


class SourceSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    x: float
    y: float
    z: float
    amplitude: float
    frequency_hz: float
    direction: list[float]

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("Source direction must have exactly 3 components.")
        return value


class ProbeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: float
    y: float
    z: float


class DomainSizeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    lx: float
    ly: float
    lz: float


class DomainResolutionSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    nx: int
    ny: int
    nz: int


class BoundaryConditionsSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    left: str
    right: str
    top: str
    bottom: str
    front: str | None = None
    back: str | None = None


class DomainSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    size: DomainSizeSchema
    resolution: DomainResolutionSchema
    boundary_conditions: BoundaryConditionsSchema


class PINNPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    medium: MediumSchema
    scenario: ScenarioSchema
    source: SourceSchema
    probe: ProbeSchema
    domain: DomainSchema
    representation: str
    routing_hint: str
