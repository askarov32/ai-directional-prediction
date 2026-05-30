"""Adapter tests: remote payload (any of 3 shapes) -> v2 normalised."""
from __future__ import annotations

import math

import pytest

from app.infrastructure.adapters.remote_response_schema_v2 import (
    NormalizedRemotePayloadV2,
    parse_remote_payload,
)


# --- v2 native --------------------------------------------------------------


def test_parse_v2_native_payload():
    payload = {
        "schema_version": "2.0",
        "model_version": "pinn-v2@best",
        "prediction_raw": {
            "temperature_k": 315.2,
            "temperature_perturbation_k": 42.05,
            "displacement_m": {"u": 1.2e-6, "v": 3.0e-7},
            "travel_time_s": 0.000186,
            "response_magnitude_score": 0.73,
        },
        "optional_outputs": {
            "field_summary": {
                "max_temperature_k": 318.4,
                "max_displacement_m": 1.5e-6,
                "max_temperature_perturbation_k": 25.0,
            },
            "confidence_score": 0.91,
            "field_grid": {"type": "rect_2d", "channels": {}},
            "field_sources": {"temperature_k": "direct_model_output"},
            "available_fields": ["temperature_k"],
            "missing_fields": ["stress_von_mises_pa"],
        },
        "diagnostics": {
            "fallback_used": False,
            "fallback_reason": None,
            "warnings": ["cpu-only"],
        },
    }
    out = parse_remote_payload(payload)
    assert out.raw_shape == "v2"
    assert out.temperature_k == pytest.approx(315.2)
    assert out.temperature_perturbation_k == pytest.approx(42.05)
    assert out.displacement_u_m == pytest.approx(1.2e-6)
    assert out.displacement_v_m == pytest.approx(3.0e-7)
    assert out.travel_time_s == pytest.approx(0.000186)
    assert out.response_magnitude_score == pytest.approx(0.73)
    assert out.max_displacement_m == pytest.approx(1.5e-6)
    assert out.max_temperature_perturbation_k == pytest.approx(25.0)
    assert out.confidence_score == pytest.approx(0.91)
    assert out.field_grid == {"type": "rect_2d", "channels": {}}
    assert out.field_summary["max_temperature_k"] == pytest.approx(318.4)
    assert out.field_sources == {"temperature_k": "direct_model_output"}
    assert out.available_fields == ["temperature_k"]
    assert out.missing_fields == ["stress_von_mises_pa"]
    assert out.model_version == "pinn-v2@best"
    assert out.fallback_used is False
    assert out.warnings == ["cpu-only"]


# --- v1 nested (current FNO) -----------------------------------------------


def test_parse_v1_nested_fno_payload_converts_ms_to_seconds():
    payload = {
        "prediction": {
            "direction_vector": [1.0, 0.0, 0.0],
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "magnitude": 0.035,
            "wave_type": "fno_checkpoint_inference",
            "travel_time_ms": 0.186,
        },
        "field_summary": {
            "max_displacement": 1.5e-6,
            "max_temperature_perturbation": 25.0,
        },
        "model_version": "fno-baseline@best",
        "diagnostics": {"fallback_used": False, "warnings": []},
    }
    out = parse_remote_payload(payload)
    assert out.raw_shape == "v1_nested"
    assert out.travel_time_s == pytest.approx(0.000186)
    assert out.max_displacement_m == pytest.approx(1.5e-6)
    assert out.max_temperature_perturbation_k == pytest.approx(25.0)
    assert out.temperature_k is None  # FNO doesn't expose direct fields in v1
    assert out.displacement_u_m is None
    assert out.displacement_v_m is None
    assert out.response_magnitude_score == pytest.approx(0.035)
    assert out.model_version == "fno-baseline@best"


# --- v1 flat (current PINN / MGN / Transformer) ---------------------------


def test_parse_v1_flat_pinn_payload_uses_model_outputs():
    payload = {
        "direction_vector": [0.977, 0.211, 0.0],
        "azimuth_deg": 12.187,
        "elevation_deg": 0.0,
        "magnitude": 0.035051,
        "wave_type": "physics_informed",
        "travel_time_ms": 0.092988,
        "max_displacement": 1.2009e-5,
        "max_temperature_perturbation": 0.052487,
        "model_version": "pinn-baseline@best_model.pth",
        "model_outputs": {
            "feature_names": [
                "temperature_k",
                "disp_x",
                "disp_y",
                "disp_z",
            ],
            "values": [293.149932, -1.1e-8, -1.2e-8, 2.2e-5],
        },
    }
    out = parse_remote_payload(payload)
    assert out.raw_shape == "v1_flat"
    assert out.temperature_k == pytest.approx(293.149932)
    assert out.displacement_u_m == pytest.approx(-1.1e-8)
    assert out.displacement_v_m == pytest.approx(-1.2e-8)
    assert out.travel_time_s == pytest.approx(0.092988 / 1000.0)
    assert out.max_displacement_m == pytest.approx(1.2009e-5)
    assert out.max_temperature_perturbation_k == pytest.approx(0.052487)
    assert out.response_magnitude_score == pytest.approx(0.035051)
    assert out.fallback_used is False


def test_parse_v1_flat_with_mgn_fallback_diagnostics():
    payload = {
        "direction_vector": [1.0, 0.0, 0.0],
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "magnitude": 1.0,
        "wave_type": "dominant_p",
        "travel_time_ms": 1.5,
        "max_displacement": 0.001,
        "max_temperature_perturbation": 1.0,
        "model_version": "mgn-fallback-v1",
        "diagnostics": {
            "fallback_used": True,
            "fallback_reason": "missing_artifact",
            "warnings": ["using deterministic stub"],
        },
    }
    out = parse_remote_payload(payload)
    assert out.fallback_used is True
    assert out.fallback_reason == "missing_artifact"
    assert "using deterministic stub" in out.warnings


# --- error path ------------------------------------------------------------


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        parse_remote_payload({"foo": "bar"})


def test_non_finite_values_become_none_via_float_coercion():
    payload = {
        "direction_vector": [1.0, 0.0, 0.0],
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "magnitude": 0.0,
        "wave_type": "x",
        "travel_time_ms": 1.0,
        "max_displacement": None,
        "max_temperature_perturbation": None,
        "model_version": "v",
    }
    out = parse_remote_payload(payload)
    assert out.max_displacement_m is None
    assert out.max_temperature_perturbation_k is None
