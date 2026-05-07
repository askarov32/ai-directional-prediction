from __future__ import annotations


def test_health_endpoint(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "thermoelastic-direction-api"}


def test_ready_endpoint_returns_readiness_checks(client):
    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["media_catalog"]["ready"] is True
    assert any(item["id"] == "meshgraphnet" and item["ready"] is True for item in payload["checks"]["models"])


def test_request_id_header_is_preserved(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "demo-request-id"})

    assert response.headers["X-Request-ID"] == "demo-request-id"


def test_media_endpoint_returns_catalog(client):
    response = client.get("/api/v1/media")

    assert response.status_code == 200
    media = response.json()
    assert len(media) >= 3
    assert any(item["id"] == "sandstone_medium" for item in media)


def test_models_endpoint_returns_supported_models(client):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()}
    assert {"meshgraphnet", "fno", "pinn"}.issubset(model_ids)


def test_prediction_happy_path_uses_mocked_model_client(client, prediction_payload):
    response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "meshgraphnet"
    assert payload["medium"]["id"] == "sandstone_medium"
    assert payload["prediction"]["direction_vector"] == [0.82, 0.57, 0.0]
    assert payload["prediction"]["wave_type"] == "dominant_p"
    assert payload["field_summary"]["max_displacement"] == 0.0032
    assert payload["meta"]["model_version"] == "fake-meshgraphnet-v1"


def test_invalid_prediction_payload_returns_validation_error(client, prediction_payload):
    prediction_payload["scenario"]["pressure_mpa"] = -1.0

    response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_medium_returns_controlled_error(client, prediction_payload):
    prediction_payload["medium_id"] = "unknown_medium"

    response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MEDIUM_NOT_FOUND"


def test_unsupported_model_type_returns_validation_error(client, prediction_payload):
    prediction_payload["model"] = "unsupported_model"

    response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"


def test_temperature_out_of_range_returns_controlled_error(client, prediction_payload):
    prediction_payload["scenario"]["temperature_c"] = 999.0

    response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TEMPERATURE_OUT_OF_RANGE"


def test_malformed_remote_response_returns_controlled_error(client_factory, prediction_payload):
    malformed_remote_payload = {
        "azimuth_deg": 34.7,
        "elevation_deg": 0.0,
        "magnitude": 1.0,
        "wave_type": "dominant_p",
        "travel_time_ms": 11.8,
        "max_displacement": 0.0032,
        "max_temperature_perturbation": 1.7,
        "model_version": "fake-meshgraphnet-v1",
    }

    with client_factory(malformed_remote_payload) as client:
        response = client.post("/api/v1/predictions", json=prediction_payload)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "MALFORMED_MODEL_RESPONSE"
