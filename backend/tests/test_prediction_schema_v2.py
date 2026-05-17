"""Unit tests for the v2 Pydantic request/response schemas.

Cover the contract guarantees in docs/api-contract-v2.md §2 and §4:
- minimal v2 request validates;
- locked fields (reference_temperature_k, source_temperature_k,
  frequency_hz, domain.size) are rejected via extra="forbid";
- coordinates outside the 1 m x 1 m domain are rejected;
- source == probe is rejected;
- 3D requests are rejected via Literal[2].
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.prediction import (
    PredictionRequestV2Schema,
)


def _minimal_v2_payload(**overrides) -> dict:
    payload = {
        "schema_version": "2.0",
        "model": "pinn",
        "medium_id": "sandstone_medium",
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


def test_minimal_v2_payload_validates():
    req = PredictionRequestV2Schema.model_validate(_minimal_v2_payload())
    entity = req.to_entity()
    assert entity.medium_id == "sandstone_medium"
    assert entity.geometry.dimension == 2
    assert entity.observation.time_s == pytest.approx(0.1)
    # thermal_state is set from locked invariants
    assert entity.thermal_state.reference_temperature_k == pytest.approx(273.15)
    assert entity.thermal_state.source_temperature_k == pytest.approx(1500.0)
    assert entity.thermal_state.theta_k == pytest.approx(1226.85)


def test_scenario_block_is_optional_and_defaults_apply():
    payload = _minimal_v2_payload()
    payload.pop("scenario")
    req = PredictionRequestV2Schema.model_validate(payload)
    assert req.scenario.thermal_source_type == "point"
    assert req.scenario.mechanical_constraint == "free"
    assert req.scenario.boundary_condition_type == "prototype_simplified"


@pytest.mark.parametrize(
    "extra_key, extra_value",
    [
        ("thermal_state", {"reference_temperature_k": 290.0}),
        ("frequency_hz", 25.0),
        ("source", {"x": 0.1, "y": 0.1, "z": 0, "amplitude": 1, "frequency_hz": 50, "type": "thermal_pulse", "direction": [1, 0, 0]}),
        ("amplitude", 1.0),
    ],
)
def test_request_rejects_v1_legacy_fields(extra_key, extra_value):
    payload = _minimal_v2_payload(**{extra_key: extra_value})
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_geometry_rejects_3d_dimension():
    payload = _minimal_v2_payload()
    payload["geometry"]["dimension"] = 3
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_geometry_rejects_out_of_domain_coords():
    payload = _minimal_v2_payload()
    payload["geometry"]["probe"]["x_m"] = 1.5
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_geometry_rejects_coincident_source_probe():
    payload = _minimal_v2_payload()
    payload["geometry"]["probe"] = {"x_m": 0.2, "y_m": 0.5}
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_observation_rejects_non_positive_time():
    payload = _minimal_v2_payload()
    payload["observation"]["time_s"] = 0.0
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_extra_unknown_field_rejected():
    payload = _minimal_v2_payload(foo="bar")
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_scenario_rejects_unknown_mechanical_constraint():
    payload = _minimal_v2_payload()
    payload["scenario"]["mechanical_constraint"] = "elastic_clamp"
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)


def test_schema_version_must_be_exactly_2_0():
    payload = _minimal_v2_payload()
    payload["schema_version"] = "1.0"
    with pytest.raises(ValidationError):
        PredictionRequestV2Schema.model_validate(payload)
