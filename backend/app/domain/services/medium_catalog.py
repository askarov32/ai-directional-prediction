from __future__ import annotations

from app.core.exceptions import DomainValidationError, ResourceNotFoundError
from app.domain.entities.medium import Medium
from app.domain.entities.prediction import Scenario
from app.domain.ports import MediumRepositoryPort


class MediumCatalogService:
    def __init__(self, repository: MediumRepositoryPort) -> None:
        self.repository = repository

    def list_media(self) -> list[Medium]:
        return self.repository.list_media()

    def get_medium(self, medium_id: str) -> Medium:
        medium = self.repository.get_by_id(medium_id)
        if medium is None:
            raise ResourceNotFoundError(
                code="MEDIUM_NOT_FOUND",
                message=f"Unknown medium_id: {medium_id}",
                details={"medium_id": medium_id},
            )
        return medium

    def validate_scenario_ranges(self, medium: Medium, scenario: Scenario) -> None:
        min_temperature, max_temperature = medium.ranges.temperature_c
        if not (min_temperature <= scenario.temperature_c <= max_temperature):
            raise DomainValidationError(
                code="TEMPERATURE_OUT_OF_RANGE",
                message=f"Temperature {scenario.temperature_c}C is outside the allowed range for {medium.name}",
                details={
                    "medium_id": medium.id,
                    "allowed_range": [min_temperature, max_temperature],
                    "received": scenario.temperature_c,
                },
            )

        min_pressure, max_pressure = medium.ranges.pressure_mpa
        if not (min_pressure <= scenario.pressure_mpa <= max_pressure):
            raise DomainValidationError(
                code="PRESSURE_OUT_OF_RANGE",
                message=f"Pressure {scenario.pressure_mpa} MPa is outside the allowed range for {medium.name}",
                details={
                    "medium_id": medium.id,
                    "allowed_range": [min_pressure, max_pressure],
                    "received": scenario.pressure_mpa,
                },
            )
