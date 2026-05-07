from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_predict_direction_use_case, get_prediction_router
from app.domain.entities.medium import Medium, MediumMetadata, MediumProperties, MediumRanges
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.entities.prediction import (
    BoundaryConditions,
    Domain,
    DomainResolution,
    DomainSize,
    Probe,
    Scenario,
    Source,
)
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

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "direction_vector": [0.82, 0.57, 0.0],
            "azimuth_deg": 34.7,
            "elevation_deg": 0.0,
            "magnitude": 1.0,
            "wave_type": "dominant_p",
            "travel_time_ms": 11.8,
            "max_displacement": 0.0032,
            "max_temperature_perturbation": 1.7,
            "model_version": "fake-meshgraphnet-v1",
        }

    async def predict(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse:
        return RemotePredictionResponse(
            service_name="MeshGraphNet",
            payload=self.payload,
            latency_ms=12,
        )

    async def readiness(self) -> dict[str, Any]:
        return {"id": "meshgraphnet", "name": "MeshGraphNet", "ready": True, "status": "ready"}

    def descriptor(self) -> dict[str, str]:
        return {"id": "meshgraphnet", "name": "MeshGraphNet", "status": "configured"}


@pytest.fixture
def enriched_request() -> EnrichedPredictionRequest:
    medium = Medium(
        id="sandstone_medium",
        name="Sandstone (medium)",
        category="sedimentary",
        properties=MediumProperties(
            rho=2684.0,
            porosity_total=0.34,
            porosity_effective=0.27,
            vp=6.17,
            vs=3.2,
            thermal_conductivity=2.5,
            heat_capacity=850.0,
            thermal_expansion=0.000012,
        ),
        ranges=MediumRanges(temperature_c=(-20.0, 300.0), pressure_mpa=(0.1, 1500.0)),
        metadata=MediumMetadata(source="test"),
    )
    return EnrichedPredictionRequest(
        model=ModelType.MESHGRAPHNET,
        medium=medium,
        scenario=Scenario(temperature_c=120.0, pressure_mpa=35.0, time_ms=12.0),
        source=Source(
            type="thermal_pulse",
            x=0.15,
            y=0.4,
            z=0.0,
            amplitude=1.0,
            frequency_hz=50.0,
            direction=(1.0, 0.0, 0.0),
        ),
        probe=Probe(x=0.7, y=0.55, z=0.0),
        domain=Domain(
            type="rect_2d",
            size=DomainSize(lx=1.0, ly=1.0, lz=0.0),
            resolution=DomainResolution(nx=128, ny=128, nz=1),
            boundary_conditions=BoundaryConditions(
                left="fixed",
                right="free",
                top="insulated",
                bottom="insulated",
            ),
        ),
    )


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
def client_factory() -> Callable[[dict[str, Any] | None], Generator[TestClient, None, None]]:
    @contextmanager
    def build_client(remote_payload: dict[str, Any] | None = None) -> Generator[TestClient, None, None]:
        repository = MediaRepository(CATALOG_PATH)
        medium_catalog = MediumCatalogService(repository)
        prediction_router = PredictionRouter([FakeMeshGraphNetClient(remote_payload)])
        use_case = PredictDirectionUseCase(
            medium_catalog=medium_catalog,
            prediction_router=prediction_router,
            response_normalizer=ResponseNormalizer(),
        )

        app.dependency_overrides[get_predict_direction_use_case] = lambda: use_case
        app.dependency_overrides[get_prediction_router] = lambda: prediction_router
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    return build_client


@pytest.fixture
def client(client_factory) -> Generator[TestClient, None, None]:
    with client_factory() as test_client:
        yield test_client
