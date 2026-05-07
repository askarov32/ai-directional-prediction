from __future__ import annotations

from app.domain.entities.prediction import EnrichedPredictionRequest, UnifiedPredictionRequest
from app.domain.ports import PredictionResponseNormalizerPort
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.services.prediction_router import PredictionRouter


class PredictDirectionUseCase:
    def __init__(
        self,
        medium_catalog: MediumCatalogService,
        prediction_router: PredictionRouter,
        response_normalizer: PredictionResponseNormalizerPort,
    ) -> None:
        self.medium_catalog = medium_catalog
        self.prediction_router = prediction_router
        self.response_normalizer = response_normalizer

    async def execute(self, request: UnifiedPredictionRequest) -> dict:
        medium = self.medium_catalog.get_medium(request.medium_id)
        self.medium_catalog.validate_scenario_ranges(medium, request.scenario)

        enriched_request = EnrichedPredictionRequest(
            model=request.model,
            medium=medium,
            scenario=request.scenario,
            source=request.source,
            probe=request.probe,
            domain=request.domain,
        )

        remote_response = await self.prediction_router.route(enriched_request)
        return self.response_normalizer.normalize(enriched_request, remote_response)
