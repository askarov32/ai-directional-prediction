from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_predict_direction_use_case
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.services.prediction_router import PredictionRouter
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.infrastructure.adapters.response_normalizer import ResponseNormalizer
from app.infrastructure.repositories.media_repository import MediaRepository
from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "backend" / "data" / "media" / "catalog.json"


class FakeMeshGraphNetClient:
    model_type = ModelType.MESHGRAPHNET

    async def predict(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse:
        return RemotePredictionResponse(
            service_name="MeshGraphNet",
            payload={
                "direction_vector": [0.82, 0.57, 0.0],
                "azimuth_deg": 34.7,
                "elevation_deg": 0.0,
                "magnitude": 1.0,
                "wave_type": "dominant_p",
                "travel_time_ms": 11.8,
                "max_displacement": 0.0032,
                "max_temperature_perturbation": 1.7,
                "model_version": "fake-meshgraphnet-v1",
            },
            latency_ms=12,
        )

    def descriptor(self) -> dict[str, str]:
        return {"id": "meshgraphnet", "name": "MeshGraphNet", "status": "configured"}


@pytest.fixture
def prediction_payload() -> dict:
    return {
        "model": "meshgraphnet",
        "medium_id": "sandstone_medium",
        "scenario": {
            "temperature_c": 120.0,
            "pressure_mpa": 35.0,
            "time_ms": 12.0,
        },
        "source": {
            "type": "thermal_pulse",
            "x": 0.15,
            "y": 0.4,
            "z": 0.0,
            "amplitude": 1.0,
            "frequency_hz": 50.0,
            "direction": [1.0, 0.0, 0.0],
        },
        "probe": {
            "x": 0.7,
            "y": 0.55,
            "z": 0.0,
        },
        "domain": {
            "type": "rect_2d",
            "size": {
                "lx": 1.0,
                "ly": 1.0,
                "lz": 0.0,
            },
            "resolution": {
                "nx": 128,
                "ny": 128,
                "nz": 1,
            },
            "boundary_conditions": {
                "left": "fixed",
                "right": "free",
                "top": "insulated",
                "bottom": "insulated",
            },
        },
    }


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    repository = MediaRepository(CATALOG_PATH)
    medium_catalog = MediumCatalogService(repository)
    prediction_router = PredictionRouter([FakeMeshGraphNetClient()])
    use_case = PredictDirectionUseCase(
        medium_catalog=medium_catalog,
        prediction_router=prediction_router,
        response_normalizer=ResponseNormalizer(),
    )

    app.dependency_overrides[get_predict_direction_use_case] = lambda: use_case
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
