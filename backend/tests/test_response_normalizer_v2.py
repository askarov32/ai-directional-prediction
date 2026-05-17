"""Tests for response_normalizer_v2.normalize_with_geometry.

Verifies the v2 contract guarantees in api-contract-v2.md §3:
- temporal_response.travel_time_s is required and populated regardless
  of which remote shape arrived;
- field_summary carries v1 backward-compat values;
- diagnostics.notes[0] is the canonical disclaimer;
- confidence/strain/stress are always null in v2;
- request_id is stamped per call.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.entities.medium import (
    MediumMetadataV2,
    MediumPropertiesV2,
    MediumV2,
)
from app.domain.entities.prediction import (
    Geometry2D,
    ObservationV2,
    Point2D,
    ScenarioPrototypeV2,
    ThermalStateV2,
    UnifiedPredictionRequestV2,
)
from app.domain.enums.model_type import ModelType
from app.domain.services.derived_quantities import (
    REFERENCE_TEMPERATURE_K,
    SOURCE_TEMPERATURE_K,
)
from app.infrastructure.adapters.response_normalizer_v2 import (
    DISCLAIMER,
    normalize_with_geometry,
)
from app.schemas.prediction import PredictionResponseV2Schema


def _make_request(model: ModelType = ModelType.PINN) -> UnifiedPredictionRequestV2:
    return UnifiedPredictionRequestV2(
        model=model,
        medium_id="granite",
        geometry=Geometry2D(
            dimension=2,
            source=Point2D(x_m=0.2, y_m=0.5),
            probe=Point2D(x_m=0.8, y_m=0.5),
        ),
        observation=ObservationV2(time_s=0.1),
        scenario=ScenarioPrototypeV2(
            thermal_source_type="point",
            mechanical_constraint="free",
            boundary_condition_type="prototype_simplified",
        ),
        thermal_state=ThermalStateV2(
            reference_temperature_k=REFERENCE_TEMPERATURE_K,
            source_temperature_k=SOURCE_TEMPERATURE_K,
        ),
    )


def _make_medium() -> MediumV2:
    return MediumV2(
        id="granite",
        name="Granite",
        category="igneous intrusive",
        thermoelastic_supported=True,
        properties=MediumPropertiesV2(
            rho_kg_m3=2650.0, vp_m_s=5850.0, vs_m_s=3400.0,
            young_modulus_pa=7.6e10, poisson_ratio=0.245,
            shear_modulus_pa=3.0e10, bulk_modulus_pa=5.0e10,
            lame_lambda_pa=2.9e10,
            thermal_conductivity_w_mk=2.5,
            heat_capacity_j_kgk=850.0,
            volumetric_heat_capacity_j_m3k=2.25e6,
            thermal_expansion_1_k=7.9e-6,
            thermoelastic_gamma_pa_k=1.18e6,
            porosity_summary="total 45.5",
        ),
        metadata=MediumMetadataV2(
            source_table="combined_geological_media_parameters.csv",
            value_type="mixed",
            source_files="",
            notes="",
        ),
    )


# --- v1 flat (PINN-style) → v2 -------------------------------------------


def test_normalizer_lifts_v1_flat_pinn_into_v2():
    remote = {
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
    out = normalize_with_geometry(
        _make_request(),
        _make_medium(),
        remote,
        route="/predict",
        inference_time_ms=2.5,
    )

    assert out["schema_version"] == "2.0"
    assert out["status"] == "ok"
    assert out["model"]["name"] == "pinn"
    assert out["model"]["version"] == "pinn-baseline@best"
    assert out["model"]["inference_time_ms"] == pytest.approx(2.5)
    assert out["material"]["id"] == "granite"
    # temporal_response is REQUIRED in v2
    assert out["prediction"]["temporal_response"]["travel_time_s"] == pytest.approx(
        0.0001
    )
    # displacement components survived
    assert out["prediction"]["displacement"]["components_m"]["u"] == pytest.approx(
        -1.1e-8
    )
    assert out["prediction"]["displacement"]["components_m"]["v"] == pytest.approx(
        -1.2e-8
    )
    # backward-compat field_summary
    assert out["optional_outputs"]["field_summary"]["max_displacement_m"] == pytest.approx(
        1.2e-5
    )
    # geometry derived
    assert out["geometry"]["distance_m"] == pytest.approx(0.6)
    assert out["geometry"]["azimuth_deg"] == pytest.approx(0.0)
    # confidence/strain/stress always null in v2
    assert out["optional_outputs"]["confidence_score"] is None
    assert out["optional_outputs"]["strain"] is None
    assert out["optional_outputs"]["stress"] is None
    # disclaimer in diagnostics.notes
    assert out["diagnostics"]["notes"][0] == DISCLAIMER


# --- v1 nested (FNO) → v2 -------------------------------------------------


def test_normalizer_lifts_v1_nested_fno_into_v2():
    remote = {
        "prediction": {
            "direction_vector": [0.8, 0.6, 0.0],
            "azimuth_deg": 36.87,
            "elevation_deg": 0.0,
            "magnitude": 0.4,
            "wave_type": "fno_checkpoint_inference",
            "travel_time_ms": 0.5,
        },
        "field_summary": {
            "max_displacement": 1.5e-6,
            "max_temperature_perturbation": 25.0,
        },
        "model_version": "fno-baseline@best",
        "diagnostics": {"fallback_used": False, "warnings": []},
    }
    out = normalize_with_geometry(
        _make_request(model=ModelType.FNO),
        _make_medium(),
        remote,
        route="/predict",
        inference_time_ms=3.2,
    )
    assert out["prediction"]["temporal_response"]["travel_time_s"] == pytest.approx(
        0.0005
    )
    assert out["prediction"]["directional_response"][
        "response_magnitude_score"
    ] == pytest.approx(0.4)
    assert out["prediction"]["displacement"]["components_m"]["u"] is None
    assert out["prediction"]["displacement"]["components_m"]["v"] is None


# --- v2 native (target) -> v2 --------------------------------------------


def test_normalizer_passes_v2_native_through_intact():
    remote = {
        "schema_version": "2.0",
        "model_version": "pinn-v2@best",
        "prediction_raw": {
            "temperature_k": 315.2,
            "displacement_m": {"u": 1.2e-6, "v": 3.0e-7},
            "travel_time_s": 0.000186,
            "response_magnitude_score": 0.73,
        },
        "optional_outputs": {
            "field_summary": {
                "max_displacement_m": 1.5e-6,
                "max_temperature_perturbation_k": 25.0,
            },
        },
        "diagnostics": {
            "fallback_used": False,
            "fallback_reason": None,
            "warnings": [],
        },
    }
    out = normalize_with_geometry(
        _make_request(),
        _make_medium(),
        remote,
        route="/predict",
        inference_time_ms=2.1,
    )
    assert out["prediction"]["thermal"]["temperature_k"]["value"] == pytest.approx(
        315.2
    )
    assert out["prediction"]["temporal_response"]["travel_time_s"] == pytest.approx(
        0.000186
    )


# --- fallback path -------------------------------------------------------


def test_normalizer_surfaces_fallback_diagnostics():
    remote = {
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
            "warnings": ["stub used"],
        },
    }
    out = normalize_with_geometry(
        _make_request(model=ModelType.MESHGRAPHNET),
        _make_medium(),
        remote,
        route="/predict",
        inference_time_ms=1.5,
    )
    assert out["model"]["fallback_used"] is True
    assert out["model"]["fallback_reason"] == "missing_artifact"
    assert out["diagnostics"]["fallback_used"] is True
    assert "stub used" in out["diagnostics"]["warnings"]


# --- Pydantic round-trip ------------------------------------------------


def test_normalised_dict_validates_as_v2_response_schema():
    remote = {
        "direction_vector": [1.0, 0.0, 0.0],
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "magnitude": 0.035,
        "wave_type": "physics_informed",
        "travel_time_ms": 0.1,
        "max_displacement": 1.2e-5,
        "max_temperature_perturbation": 0.05,
        "model_version": "pinn-baseline@best",
    }
    out = normalize_with_geometry(
        _make_request(),
        _make_medium(),
        remote,
        route="/predict",
        inference_time_ms=2.5,
    )
    # Must round-trip cleanly through the Pydantic response schema
    PredictionResponseV2Schema.model_validate(out)


def test_request_id_is_unique_per_call():
    remote = {
        "direction_vector": [1.0, 0.0, 0.0],
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "magnitude": 0.1,
        "wave_type": "x",
        "travel_time_ms": 0.1,
        "max_displacement": 1e-5,
        "max_temperature_perturbation": 0.1,
        "model_version": "x",
    }
    a = normalize_with_geometry(
        _make_request(), _make_medium(), remote,
        route="/predict", inference_time_ms=None,
    )
    b = normalize_with_geometry(
        _make_request(), _make_medium(), remote,
        route="/predict", inference_time_ms=None,
    )
    assert a["request_id"] != b["request_id"]
