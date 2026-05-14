from __future__ import annotations

import math

import pytest

from app.core.exceptions import MalformedRemoteResponseError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.infrastructure.adapters.response_normalizer import ResponseNormalizer


@pytest.fixture
def flat_remote_payload() -> dict:
    return {
        "direction_vector": [0.82, 0.57, 0.0],
        "azimuth_deg": 34.7,
        "elevation_deg": 0.0,
        "magnitude": 1.0,
        "wave_type": "dominant_p",
        "travel_time_ms": 11.8,
        "max_displacement": 0.0032,
        "max_temperature_perturbation": 1.7,
        "model_version": "mock-meshgraphnet-v1",
    }


def normalize_payload(enriched_request: EnrichedPredictionRequest, payload: dict) -> dict:
    return ResponseNormalizer().normalize(
        enriched_request,
        RemotePredictionResponse(service_name="MeshGraphNet", payload=payload, latency_ms=14),
    )


def test_normalizer_accepts_valid_flat_remote_response(enriched_request, flat_remote_payload):
    result = normalize_payload(enriched_request, flat_remote_payload)

    assert result["prediction"]["direction_vector"] == [0.82, 0.57, 0.0]
    assert result["prediction"]["magnitude"] == 1.0
    assert result["prediction"]["travel_time_ms"] == 11.8
    assert result["field_summary"]["max_temperature_perturbation"] == 1.7
    assert result["meta"]["model_version"] == "mock-meshgraphnet-v1"


def test_normalizer_accepts_valid_nested_remote_response(enriched_request, flat_remote_payload):
    payload = {
        "prediction": {
            key: flat_remote_payload[key]
            for key in [
                "direction_vector",
                "azimuth_deg",
                "elevation_deg",
                "magnitude",
                "wave_type",
                "travel_time_ms",
            ]
        },
        "field_summary": {
            "max_displacement": flat_remote_payload["max_displacement"],
            "max_temperature_perturbation": flat_remote_payload["max_temperature_perturbation"],
        },
        "meta": {"model_version": "nested-v1"},
    }

    result = normalize_payload(enriched_request, payload)

    assert result["meta"]["model_version"] == "nested-v1"
    assert result["prediction"]["wave_type"] == "dominant_p"


def test_normalizer_accepts_fno_checkpoint_style_response(enriched_request):
    payload = {
        "prediction": {
            "direction_vector": [0.821, 0.571, 0.0],
            "azimuth_deg": 34.8,
            "elevation_deg": 0.0,
            "magnitude": 0.914321,
            "wave_type": "fno_checkpoint_inference",
            "travel_time_ms": 12.41234,
        },
        "field_summary": {
            "max_displacement": 0.0013274,
            "max_temperature_perturbation": 1.7421,
        },
        "model_version": "fno-baseline@best_model.pth",
        "diagnostics": {
            "checkpoint_loaded": True,
            "device": "cpu",
            "input_channels": ["temperature_k"],
            "output_channels": ["temperature_k", "disp_x", "disp_y", "disp_z"],
        },
    }

    result = ResponseNormalizer().normalize(
        enriched_request,
        RemotePredictionResponse(service_name="FNO", payload=payload, latency_ms=22),
    )

    assert result["prediction"]["wave_type"] == "fno_checkpoint_inference"
    assert result["prediction"]["magnitude"] == 0.9143
    assert result["prediction"]["travel_time_ms"] == 12.412
    assert result["field_summary"]["max_displacement"] == 0.001327
    assert result["meta"]["model_version"] == "fno-baseline@best_model.pth"


@pytest.mark.parametrize(
    ("mutate", "reason_fragment"),
    [
        (lambda payload: payload.pop("direction_vector"), "direction_vector"),
        (lambda payload: payload.update({"direction_vector": [1.0, 0.0]}), "direction_vector"),
        (lambda payload: payload.update({"direction_vector": [0.0, 0.0, 0.0]}), "direction_vector"),
        (lambda payload: payload.update({"magnitude": -1.0}), "magnitude"),
        (lambda payload: payload.update({"travel_time_ms": math.inf}), "travel_time_ms"),
        (lambda payload: payload.pop("max_displacement"), "max_displacement"),
        (lambda payload: payload.pop("wave_type"), "wave_type"),
        (lambda payload: payload.pop("model_version"), "model_version"),
    ],
)
def test_normalizer_rejects_malformed_flat_remote_response(
    enriched_request,
    flat_remote_payload,
    mutate,
    reason_fragment,
):
    mutate(flat_remote_payload)

    with pytest.raises(MalformedRemoteResponseError) as exc_info:
        normalize_payload(enriched_request, flat_remote_payload)

    assert reason_fragment in str(exc_info.value.details["errors"])


def test_normalizer_rejects_missing_nested_prediction(enriched_request, flat_remote_payload):
    payload = {
        "field_summary": {
            "max_displacement": flat_remote_payload["max_displacement"],
            "max_temperature_perturbation": flat_remote_payload["max_temperature_perturbation"],
        },
        "model_version": "nested-v1",
    }

    with pytest.raises(MalformedRemoteResponseError) as exc_info:
        normalize_payload(enriched_request, payload)

    assert "prediction" in str(exc_info.value.details["errors"])
