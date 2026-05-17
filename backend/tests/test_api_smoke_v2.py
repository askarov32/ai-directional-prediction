"""Route dispatch smoke tests for the v2 contract.

Verifies that POST /api/v1/predictions:
- still serves the v1 response when schema_version is missing or "1.0";
- serves a v2-shaped response when schema_version is "2.0";
- rejects v2 requests carrying legacy fields (frequency_hz, etc.) with 422.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_media_repository_v2,
    get_medium_catalog_service_v2,
    get_predict_direction_use_case,
    get_predict_direction_v2_use_case,
    get_prediction_router,
)
from app.domain.entities.prediction import (
    EnrichedPredictionRequest,
    RemotePredictionResponse,
)
from app.domain.enums.model_type import ModelType
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.services.medium_catalog_v2 import MediumCatalogServiceV2
from app.domain.services.prediction_router import PredictionRouter
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.domain.use_cases.predict_direction_v2 import PredictDirectionV2UseCase
from app.infrastructure.adapters.response_normalizer import ResponseNormalizer
from app.infrastructure.repositories.media_repository import MediaRepository
from app.infrastructure.repositories.media_repository_v2 import (
    MediaRepositoryV2,
)
from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_V1 = PROJECT_ROOT / "backend" / "data" / "media" / "catalog.json"
CATALOG_V2 = PROJECT_ROOT / "backend" / "data" / "media" / "catalog_v2.json"


class _FakePINNClient:
    """Returns a canned v1-flat payload (same shape PINN produces today)."""

    model_type = ModelType.PINN

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "direction_vector": [1.0, 0.0, 0.0],
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "magnitude": 0.035,
            "wave_type": "physics_informed",
            "travel_time_ms": 0.1,
            "max_displacement": 1.2e-5,
            "max_temperature_perturbation": 0.05,
            "model_version": "pinn-baseline@best",
            "model_outputs": {
                "feature_names": ["temperature_k", "disp_x", "disp_y", "disp_z"],
                "values": [293.15, -1.1e-8, -1.2e-8, 2.2e-5],
            },
        }

    async def predict(
        self, request: EnrichedPredictionRequest
    ) -> RemotePredictionResponse:
        return RemotePredictionResponse(
            service_name="PINN", payload=self.payload, latency_ms=3
        )

    async def readiness(self) -> dict[str, Any]:
        return {"id": "pinn", "name": "PINN", "ready": True, "status": "ready"}

    def descriptor(self) -> dict[str, str]:
        return {"id": "pinn", "name": "PINN", "status": "configured"}


@contextmanager
def _client_with_overrides() -> Generator[TestClient, None, None]:
    """Wire BOTH v1 and v2 use cases against the same fake PINN router."""
    fake_client = _FakePINNClient()
    router = PredictionRouter([fake_client])

    v1_repo = MediaRepository(CATALOG_V1)
    v1_catalog = MediumCatalogService(v1_repo)
    v1_use_case = PredictDirectionUseCase(
        medium_catalog=v1_catalog,
        prediction_router=router,
        response_normalizer=ResponseNormalizer(),
    )

    v2_repo = MediaRepositoryV2(CATALOG_V2)
    v2_catalog = MediumCatalogServiceV2(v2_repo)
    v2_use_case = PredictDirectionV2UseCase(
        medium_catalog=v2_catalog, prediction_router=router
    )

    app.dependency_overrides[get_prediction_router] = lambda: router
    app.dependency_overrides[get_predict_direction_use_case] = lambda: v1_use_case
    app.dependency_overrides[get_predict_direction_v2_use_case] = lambda: v2_use_case
    app.dependency_overrides[get_media_repository_v2] = lambda: v2_repo
    app.dependency_overrides[get_medium_catalog_service_v2] = lambda: v2_catalog

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with _client_with_overrides() as test_client:
        yield test_client


def _v2_payload(**overrides) -> dict:
    payload = {
        "schema_version": "2.0",
        "model": "pinn",
        "medium_id": "granite",
        "geometry": {
            "dimension": 2,
            "source": {"x_m": 0.2, "y_m": 0.5},
            "probe": {"x_m": 0.8, "y_m": 0.5},
        },
        "observation": {"time_s": 0.1},
        "scenario": {
            "thermal_source_type": "point",
            "mechanical_constraint": "free",
            "boundary_condition_type": "prototype_simplified",
        },
    }
    payload.update(overrides)
    return payload


def _v1_payload() -> dict:
    return {
        "model": "pinn",
        "medium_id": "sandstone_medium",
        "scenario": {"temperature_c": 120.0, "pressure_mpa": 35.0, "time_ms": 12.0},
        "source": {
            "type": "thermal_pulse",
            "x": 0.15, "y": 0.4, "z": 0.0,
            "amplitude": 1.0, "frequency_hz": 50.0,
            "direction": [1.0, 0.0, 0.0],
        },
        "probe": {"x": 0.7, "y": 0.55, "z": 0.0},
        "domain": {
            "type": "rect_2d",
            "size": {"lx": 1.0, "ly": 1.0, "lz": 0.0},
            "resolution": {"nx": 128, "ny": 128, "nz": 1},
            "boundary_conditions": {
                "left": "fixed", "right": "free",
                "top": "insulated", "bottom": "insulated",
            },
        },
    }


def test_route_v1_path_unchanged(client):
    """A v1-shaped request (no schema_version) returns the v1 envelope."""
    response = client.post("/api/v1/predictions", json=_v1_payload())
    assert response.status_code == 200
    body = response.json()
    # v1 has top-level fields: model, medium, prediction, field_summary, meta
    assert body["model"] == "pinn"
    assert "meta" in body
    assert "field_summary" in body
    assert "schema_version" not in body  # v1 has no schema_version


def test_route_v2_path_returns_v2_envelope(client):
    response = client.post("/api/v1/predictions", json=_v2_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "2.0"
    assert body["status"] == "ok"
    # required v2 blocks
    assert body["prediction"]["temporal_response"]["travel_time_s"] is not None
    assert "thermal" in body["prediction"]
    assert "displacement" in body["prediction"]
    assert "directional_response" in body["prediction"]
    # disclaimer is always there
    assert body["diagnostics"]["notes"][0].startswith("Prototype prediction")
    # v2-only blocks
    assert body["model"]["fallback_used"] is False
    assert body["material"]["id"] == "granite"


def test_route_v2_rejects_legacy_field(client):
    """frequency_hz at top level is a forbidden v1 hangover in v2."""
    body = _v2_payload(frequency_hz=50.0)
    response = client.post("/api/v1/predictions", json=body)
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_route_v2_rejects_unsupported_thermoelastic_material(client):
    """basalt has no alpha_1_K in catalog_v2 -> thermoelastic_supported=false."""
    body = _v2_payload(medium_id="basalt")
    response = client.post("/api/v1/predictions", json=body)
    assert response.status_code == 422 or response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "material_thermoelastic_unsupported"


def test_route_v2_rejects_out_of_domain_probe(client):
    body = _v2_payload()
    body["geometry"]["probe"]["x_m"] = 1.5
    response = client.post("/api/v1/predictions", json=body)
    assert response.status_code == 422


def test_route_v2_rejects_3d_dimension(client):
    body = _v2_payload()
    body["geometry"]["dimension"] = 3
    response = client.post("/api/v1/predictions", json=body)
    assert response.status_code == 422
