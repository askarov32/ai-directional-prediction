from __future__ import annotations

import pytest

from app.domain.entities.medium import Medium
from app.domain.entities.prediction import (
    EnrichedPredictionRequest,
    RemotePredictionResponse,
    UnifiedPredictionRequest,
)
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.use_cases.predict_direction import PredictDirectionUseCase


class FakeMediumCatalog:
    def __init__(self, medium: Medium) -> None:
        self.medium = medium
        self.validated_request: tuple[Medium, object] | None = None

    def get_medium(self, medium_id: str) -> Medium:
        assert medium_id == self.medium.id
        return self.medium

    def validate_scenario_ranges(self, medium: Medium, scenario: object) -> None:
        self.validated_request = (medium, scenario)


class FakePredictionRouter:
    def __init__(self) -> None:
        self.routed_request: EnrichedPredictionRequest | None = None

    async def route(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse:
        self.routed_request = request
        return RemotePredictionResponse(
            service_name="MeshGraphNet",
            payload={
                "direction_vector": [1.0, 0.0, 0.0],
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "magnitude": 1.0,
                "wave_type": "dominant_p",
                "travel_time_ms": 10.0,
                "max_displacement": 0.001,
                "max_temperature_perturbation": 0.5,
                "model_version": "fake-v1",
            },
            latency_ms=5,
        )


class FakeNormalizer:
    def __init__(self) -> None:
        self.normalized_request: EnrichedPredictionRequest | None = None
        self.normalized_remote: RemotePredictionResponse | None = None

    def normalize(self, request: EnrichedPredictionRequest, remote: RemotePredictionResponse) -> dict:
        self.normalized_request = request
        self.normalized_remote = remote
        return {
            "model": request.model.value,
            "medium": request.medium.summary(),
            "prediction": {"direction_vector": [1.0, 0.0, 0.0]},
        }


@pytest.mark.anyio
async def test_predict_direction_use_case_resolves_routes_and_normalizes(enriched_request):
    medium_catalog = FakeMediumCatalog(enriched_request.medium)
    prediction_router = FakePredictionRouter()
    normalizer = FakeNormalizer()
    use_case = PredictDirectionUseCase(
        medium_catalog=medium_catalog,  # type: ignore[arg-type]
        prediction_router=prediction_router,  # type: ignore[arg-type]
        response_normalizer=normalizer,
    )
    request = UnifiedPredictionRequest(
        model=enriched_request.model,
        medium_id=enriched_request.medium.id,
        scenario=enriched_request.scenario,
        source=enriched_request.source,
        probe=enriched_request.probe,
        domain=enriched_request.domain,
    )

    result = await use_case.execute(request)

    assert result["model"] == "meshgraphnet"
    assert result["medium"]["id"] == "sandstone_medium"
    assert medium_catalog.validated_request == (enriched_request.medium, enriched_request.scenario)
    assert prediction_router.routed_request is not None
    assert prediction_router.routed_request.medium == enriched_request.medium
    assert normalizer.normalized_request == prediction_router.routed_request
    assert normalizer.normalized_remote is not None


def test_medium_catalog_service_depends_on_repository_port(enriched_request):
    class InMemoryMediumRepository:
        def list_media(self) -> list[Medium]:
            return [enriched_request.medium]

        def get_by_id(self, medium_id: str) -> Medium | None:
            return enriched_request.medium if medium_id == enriched_request.medium.id else None

    service = MediumCatalogService(InMemoryMediumRepository())

    assert service.get_medium("sandstone_medium") == enriched_request.medium
    assert service.list_media() == [enriched_request.medium]
