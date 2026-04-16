from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MediumProperties:
    rho: float
    porosity_total: float
    porosity_effective: float
    vp: float
    vs: float
    thermal_conductivity: float
    heat_capacity: float
    thermal_expansion: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rho": self.rho,
            "porosity_total": self.porosity_total,
            "porosity_effective": self.porosity_effective,
            "vp": self.vp,
            "vs": self.vs,
            "thermal_conductivity": self.thermal_conductivity,
            "heat_capacity": self.heat_capacity,
            "thermal_expansion": self.thermal_expansion,
        }


@dataclass(frozen=True)
class MediumRanges:
    temperature_c: tuple[float, float]
    pressure_mpa: tuple[float, float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "temperature_c": [self.temperature_c[0], self.temperature_c[1]],
            "pressure_mpa": [self.pressure_mpa[0], self.pressure_mpa[1]],
        }


@dataclass(frozen=True)
class MediumMetadata:
    source: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "notes": self.notes}


@dataclass(frozen=True)
class Medium:
    id: str
    name: str
    category: str
    properties: MediumProperties
    ranges: MediumRanges
    metadata: MediumMetadata

    def summary(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "properties": self.properties.to_dict(),
            "ranges": self.ranges.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
