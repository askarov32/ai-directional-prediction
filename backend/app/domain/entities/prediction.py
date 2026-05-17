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


# ---------------------------------------------------------------------------
# v2 contract (api_contract_v2.md). Additive — v1 dataclasses above are not
# touched so existing tests and clients keep working.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalStateV2:
    """All thermal parameters are training-data invariants in v2.

    The contract locks reference_temperature_k = 273.15 K and
    source_temperature_k = 1500 K, so theta_k = 1226.85 K. The dataclass
    keeps the explicit values for traceability inside the pipeline.
    """

    reference_temperature_k: float
    source_temperature_k: float

    @property
    def theta_k(self) -> float:
        return self.source_temperature_k - self.reference_temperature_k

    def to_dict(self) -> dict[str, float]:
        return {
            "reference_temperature_k": self.reference_temperature_k,
            "source_temperature_k": self.source_temperature_k,
            "temperature_perturbation_k": self.theta_k,
        }


@dataclass(frozen=True)
class Point2D:
    x_m: float
    y_m: float

    def to_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}


@dataclass(frozen=True)
class Geometry2D:
    dimension: int  # always 2 in v2
    source: Point2D
    probe: Point2D

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "source": self.source.to_dict(),
            "probe": self.probe.to_dict(),
        }


@dataclass(frozen=True)
class DerivedGeometry2D:
    propagation_vector_m: tuple[float, float]
    distance_m: float
    unit_direction: tuple[float, float]
    azimuth_deg: float
    azimuth_convention: str = "atan2(dy, dx), degrees, xy-plane"

    def to_dict(self) -> dict[str, Any]:
        return {
            "propagation_vector_m": {
                "dx": self.propagation_vector_m[0],
                "dy": self.propagation_vector_m[1],
            },
            "distance_m": self.distance_m,
            "unit_direction": {
                "x": self.unit_direction[0],
                "y": self.unit_direction[1],
            },
            "azimuth_deg": self.azimuth_deg,
            "azimuth_convention": self.azimuth_convention,
        }


@dataclass(frozen=True)
class ObservationV2:
    time_s: float

    def to_dict(self) -> dict[str, float]:
        return {"time_s": self.time_s}


@dataclass(frozen=True)
class ScenarioPrototypeV2:
    """Boundary-condition template tokens.

    These do not carry numerical physics; they label which prototype
    scenario the client is requesting so prompts/checkpoints can route
    accordingly.
    """

    thermal_source_type: str
    mechanical_constraint: str
    boundary_condition_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "thermal_source_type": self.thermal_source_type,
            "mechanical_constraint": self.mechanical_constraint,
            "boundary_condition_type": self.boundary_condition_type,
        }


@dataclass(frozen=True)
class UnifiedPredictionRequestV2:
    """v2 unified request as parsed from the public API."""

    model: ModelType
    medium_id: str
    geometry: Geometry2D
    observation: ObservationV2
    scenario: ScenarioPrototypeV2
    thermal_state: ThermalStateV2  # populated by backend from locked defaults


@dataclass(frozen=True)
class NormalizedPredictionOutputV2:
    """Domain-level representation of a normalised v2 response.

    Used between the response normaliser and the API serialiser so the
    use case never speaks raw JSON.
    """

    temperature_k: float | None
    temperature_perturbation_k: float | None
    displacement_u_m: float | None
    displacement_v_m: float | None
    displacement_magnitude_m: float | None
    travel_time_s: float | None
    response_magnitude_score: float | None
    field_summary: dict[str, Any]
    fallback_used: bool
    fallback_reason: str | None
    warnings: list[str]
