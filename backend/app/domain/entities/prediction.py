from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.entities.medium import Medium
from app.domain.enums.model_type import ModelType


@dataclass(frozen=True)
class Scenario:
    temperature_c: float
    pressure_mpa: float
    time_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature_c": self.temperature_c,
            "pressure_mpa": self.pressure_mpa,
            "time_ms": self.time_ms,
        }


@dataclass(frozen=True)
class Source:
    type: str
    x: float
    y: float
    z: float
    amplitude: float
    frequency_hz: float
    direction: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "direction": list(self.direction),
        }


@dataclass(frozen=True)
class Probe:
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class DomainSize:
    lx: float
    ly: float
    lz: float

    def to_dict(self) -> dict[str, float]:
        return {"lx": self.lx, "ly": self.ly, "lz": self.lz}


@dataclass(frozen=True)
class DomainResolution:
    nx: int
    ny: int
    nz: int

    def to_dict(self) -> dict[str, int]:
        return {"nx": self.nx, "ny": self.ny, "nz": self.nz}


@dataclass(frozen=True)
class BoundaryConditions:
    left: str
    right: str
    top: str
    bottom: str
    front: str | None = None
    back: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }
        if self.front is not None:
            payload["front"] = self.front
        if self.back is not None:
            payload["back"] = self.back
        return payload


@dataclass(frozen=True)
class Domain:
    type: str
    size: DomainSize
    resolution: DomainResolution
    boundary_conditions: BoundaryConditions

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "size": self.size.to_dict(),
            "resolution": self.resolution.to_dict(),
            "boundary_conditions": self.boundary_conditions.to_dict(),
        }


@dataclass(frozen=True)
class UnifiedPredictionRequest:
    model: ModelType
    medium_id: str
    scenario: Scenario
    source: Source
    probe: Probe
    domain: Domain


@dataclass(frozen=True)
class EnrichedPredictionRequest:
    model: ModelType
    medium: Medium
    scenario: Scenario
    source: Source
    probe: Probe
    domain: Domain

    def to_shared_payload(self) -> dict[str, Any]:
        return {
            "medium": {
                **self.medium.summary(),
                "properties": self.medium.properties.to_dict(),
                "ranges": self.medium.ranges.to_dict(),
                "metadata": self.medium.metadata.to_dict(),
            },
            "scenario": self.scenario.to_dict(),
            "source": self.source.to_dict(),
            "probe": self.probe.to_dict(),
            "domain": self.domain.to_dict(),
        }


@dataclass(frozen=True)
class RemotePredictionResponse:
    service_name: str
    payload: dict[str, Any]
    latency_ms: int
