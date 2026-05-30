from __future__ import annotations

import httpx
import pytest

from app.infrastructure.clients.fno_client import FNOClient


def patch_async_client(monkeypatch, handler) -> None:
    original_async_client = httpx.AsyncClient

    def build_client(**kwargs):
        return original_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", build_client)


def test_fno_client_build_payload_adds_grid_hints(enriched_request) -> None:
    client = FNOClient(base_url="http://fno-service:9000", predict_path="/predict", timeout_seconds=10.0)

    payload = client.build_payload(enriched_request)

    assert payload["representation"] == "grid"
    assert payload["routing_hint"] == "fno"
    assert payload["requested_outputs"] == [
        "direction",
        "field_grid",
        "field_summary",
        "probe_sample",
        "diagnostics",
    ]
    assert payload["grid_policy"] == "service_default"
    assert payload["medium"]["id"] == "sandstone_medium"


@pytest.mark.anyio
async def test_fno_client_accepts_nested_checkpoint_response(monkeypatch, enriched_request) -> None:
    client = FNOClient(base_url="http://fno-service:9000", predict_path="/predict", timeout_seconds=10.0)
    nested_response = {
        "prediction": {
            "direction_vector": [0.821, 0.571, 0.0],
            "azimuth_deg": 34.8,
            "elevation_deg": 0.0,
            "magnitude": 0.914,
            "wave_type": "fno_checkpoint_inference",
            "travel_time_ms": 12.4,
        },
        "field_summary": {
            "max_displacement": 0.001327,
            "max_temperature_perturbation": 1.742,
        },
        "model_version": "fno-baseline@best_model.pth",
        "diagnostics": {
            "checkpoint_loaded": True,
            "device": "cpu",
            "input_channels": ["temperature_k"],
            "output_channels": ["temperature_k", "disp_x", "disp_y", "disp_z"],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://fno-service:9000/predict")
        return httpx.Response(200, json=nested_response)

    patch_async_client(monkeypatch, handler)

    response = await client.predict(enriched_request)

    assert response.service_name == "FNO"
    assert response.payload["prediction"]["wave_type"] == "fno_checkpoint_inference"
    assert response.payload["diagnostics"]["checkpoint_loaded"] is True
