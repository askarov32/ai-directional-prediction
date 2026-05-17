"""v2 medium catalog — refuses thermoelastically-unsupported materials."""
from __future__ import annotations

from app.core.exceptions import DomainValidationError, ResourceNotFoundError
from app.domain.entities.medium import MediumV2
from app.infrastructure.repositories.media_repository_v2 import (
    MediaRepositoryV2,
)


class MediumCatalogServiceV2:
    def __init__(self, repository: MediaRepositoryV2) -> None:
        self.repository = repository

    def list_media(self) -> list[MediumV2]:
        return self.repository.list_media()

    def get_medium(self, medium_id: str) -> MediumV2:
        medium = self.repository.get_by_id(medium_id)
        if medium is None:
            raise ResourceNotFoundError(
                code="MEDIUM_NOT_FOUND",
                message=f"Unknown medium_id: {medium_id}",
                details={"medium_id": medium_id},
            )
        return medium

    def require_thermoelastic_support(self, medium: MediumV2) -> None:
        """Raise if the material has no thermal_expansion in the source CSV.

        Catalog rule per api-contract-v2.md §4.4: materials without
        ``alpha_1_K`` are flagged ``thermoelastic_supported: false`` and
        cannot be used for thermoelastic predictions.
        """
        if not medium.thermoelastic_supported:
            raise DomainValidationError(
                code="material_thermoelastic_unsupported",
                message=(
                    f"Material '{medium.id}' has no thermal_expansion_1_k value "
                    f"in the source table; thermoelastic predictions are not "
                    f"supported for this medium."
                ),
                details={"medium_id": medium.id},
            )
