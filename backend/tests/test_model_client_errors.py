from __future__ import annotations

import httpx
import pytest

from app.core.exceptions import (
    MalformedRemoteResponseError,
    RemoteServiceHTTPError,
    RemoteServiceTimeoutError,
    RemoteServiceUnavailableError,
)
from app.domain.entities.prediction import EnrichedPredictionRequest
from app.domain.enums.model_type import ModelType
from app.infrastructure.clients.base import BaseModelClient


class StubModelClient(BaseModelClient):
    def __init__(self) -> None:
        super().__init__(
            model_type=ModelType.MESHGRAPHNET,
            service_name="MeshGraphNet",
            base_url="http://model.test",
            predict_path="/predict",
            timeout_seconds=1.0,
        )

    def build_payload(self, request: EnrichedPredictionRequest) -> dict:
        return {"ok": True}


VALID_REMOTE_RESPONSE = {
    "direction_vector": [0.82, 0.57, 0.0],
    "azimuth_deg": 34.7,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "dominant_p",
    "travel_time_ms": 11.8,
    "max_displacement": 0.0032,
    "max_temperature_perturbation": 1.7,
    "model_version": "test-v1",
}


@pytest.fixture
def model_client() -> StubModelClient:
    return StubModelClient()


def patch_async_client(monkeypatch, handler) -> None:
    original_async_client = httpx.AsyncClient

    def build_client(**kwargs):
        return original_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", build_client)


@pytest.mark.parametrize(
    ("transport_exception", "expected_exception", "expected_code"),
    [
        (httpx.ReadTimeout("timed out"), RemoteServiceTimeoutError, "MODEL_TIMEOUT"),
        (httpx.ConnectError("connection failed"), RemoteServiceUnavailableError, "MODEL_UNAVAILABLE"),
    ],
)
@pytest.mark.anyio
async def test_model_client_maps_transport_errors(
    monkeypatch,
    enriched_request,
    model_client,
    transport_exception,
    expected_exception,
    expected_code,
):
    def handler(_: httpx.Request) -> httpx.Response:
        raise transport_exception

    patch_async_client(monkeypatch, handler)

    with pytest.raises(expected_exception) as exc_info:
        await model_client.predict(enriched_request)

    assert exc_info.value.code == expected_code
    assert exc_info.value.details["model"] == "meshgraphnet"
    assert exc_info.value.details["url"] == "http://model.test/predict"


@pytest.mark.anyio
async def test_model_client_maps_non_2xx_response(monkeypatch, enriched_request, model_client):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service down"})

    patch_async_client(monkeypatch, handler)

    with pytest.raises(RemoteServiceHTTPError) as exc_info:
        await model_client.predict(enriched_request)

    assert exc_info.value.code == "MODEL_HTTP_ERROR"
    assert exc_info.value.status_code == 502
    assert exc_info.value.details["status_code"] == 503


@pytest.mark.anyio
async def test_model_client_maps_invalid_json(monkeypatch, enriched_request, model_client):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    patch_async_client(monkeypatch, handler)

    with pytest.raises(MalformedRemoteResponseError) as exc_info:
        await model_client.predict(enriched_request)

    assert exc_info.value.code == "MALFORMED_MODEL_RESPONSE"
    assert exc_info.value.details["reason"] == "response body is not valid JSON"


@pytest.mark.anyio
async def test_model_client_rejects_non_object_json(monkeypatch, enriched_request, model_client):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[VALID_REMOTE_RESPONSE])

    patch_async_client(monkeypatch, handler)

    with pytest.raises(MalformedRemoteResponseError) as exc_info:
        await model_client.predict(enriched_request)

    assert exc_info.value.code == "MALFORMED_MODEL_RESPONSE"
    assert exc_info.value.details["reason"] == "response is not a JSON object"


@pytest.mark.anyio
async def test_model_client_returns_remote_prediction_response(monkeypatch, enriched_request, model_client):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=VALID_REMOTE_RESPONSE)

    patch_async_client(monkeypatch, handler)

    response = await model_client.predict(enriched_request)

    assert response.service_name == "MeshGraphNet"
    assert response.payload == VALID_REMOTE_RESPONSE
    assert response.latency_ms >= 0
