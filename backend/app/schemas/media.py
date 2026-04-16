from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.domain.entities.medium import Medium


class MediumPropertiesSchema(BaseModel):
    rho: float
    porosity_total: float
    porosity_effective: float
    vp: float
    vs: float
    thermal_conductivity: float
    heat_capacity: float
    thermal_expansion: float


class MediumRangesSchema(BaseModel):
    temperature_c: list[float]
    pressure_mpa: list[float]


class MediumMetadataSchema(BaseModel):
    source: str
    notes: str | None = None


class MediumResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    category: str
    properties: MediumPropertiesSchema
    ranges: MediumRangesSchema
    metadata: MediumMetadataSchema

    @classmethod
    def from_entity(cls, medium: Medium) -> "MediumResponseSchema":
        return cls.model_validate(medium.to_dict())
