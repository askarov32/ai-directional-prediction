from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequestSchema


def test_prediction_schema_accepts_valid_rect_2d(prediction_payload):
    schema = PredictionRequestSchema.model_validate(prediction_payload)

    assert schema.domain.type == "rect_2d"
    assert schema.domain.resolution.nz == 1


def test_prediction_schema_accepts_valid_rect_3d(prediction_payload):
    prediction_payload["domain"] = {
        "type": "rect_3d",
        "size": {"lx": 1.0, "ly": 1.0, "lz": 0.6},
        "resolution": {"nx": 64, "ny": 64, "nz": 24},
        "boundary_conditions": {
            "left": "fixed",
            "right": "free",
            "top": "insulated",
            "bottom": "insulated",
            "front": "free",
            "back": "free",
        },
    }
    prediction_payload["source"]["z"] = 0.1
    prediction_payload["source"]["direction"] = [0.8, 0.2, 0.4]
    prediction_payload["probe"]["z"] = 0.4

    schema = PredictionRequestSchema.model_validate(prediction_payload)

    assert schema.domain.type == "rect_3d"
    assert schema.probe.z == 0.4


def test_prediction_schema_rejects_invalid_source_coordinates(prediction_payload):
    prediction_payload["source"]["x"] = 2.0

    with pytest.raises(ValidationError, match="Invalid coordinates for source"):
        PredictionRequestSchema.model_validate(prediction_payload)


def test_prediction_schema_rejects_invalid_probe_coordinates(prediction_payload):
    prediction_payload["probe"]["y"] = 2.0

    with pytest.raises(ValidationError, match="Invalid coordinates for probe"):
        PredictionRequestSchema.model_validate(prediction_payload)


def test_prediction_schema_rejects_wrong_direction_vector_length(prediction_payload):
    prediction_payload["source"]["direction"] = [1.0, 0.0]

    with pytest.raises(ValidationError, match="Direction vector must contain exactly three values"):
        PredictionRequestSchema.model_validate(prediction_payload)


def test_prediction_schema_rejects_zero_direction_vector(prediction_payload):
    prediction_payload["source"]["direction"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValidationError, match="Direction vector magnitude must be greater than zero"):
        PredictionRequestSchema.model_validate(prediction_payload)


def test_prediction_schema_rejects_invalid_rect_2d_resolution(prediction_payload):
    prediction_payload["domain"]["resolution"]["nz"] = 2

    with pytest.raises(ValidationError, match="resolution.nz must be 1"):
        PredictionRequestSchema.model_validate(prediction_payload)


def test_prediction_schema_rejects_negative_time(prediction_payload):
    prediction_payload["scenario"]["time_ms"] = -1.0

    with pytest.raises(ValidationError):
        PredictionRequestSchema.model_validate(prediction_payload)
