"""v2 response normaliser.

Given an enriched v2 request, the latency-tagged remote response, and
the parsed remote payload, build the dict that matches
``PredictionResponseV2Schema`` (api-contract-v2.md §3.1).

Pure function — no IO, no FastAPI, no Pydantic at this layer; the
HTTP serialiser instantiates the Pydantic schema downstream.
"""
from __future__ import annotations

from uuid import uuid4

from app.domain.entities.medium import MediumV2
from app.domain.entities.prediction import (
    DerivedGeometry2D,
    UnifiedPredictionRequestV2,
)
from app.domain.services.derived_quantities import (
    REFERENCE_TEMPERATURE_K,
    compute_derived_geometry,
    compute_displacement_magnitude_m,
    compute_theta_k,
)
from app.infrastructure.adapters.remote_response_schema_v2 import (
    NormalizedRemotePayloadV2,
    parse_remote_payload,
)


DISCLAIMER = (
    "Prototype prediction; not a field-validated thermoelastic simulation."
)


def _round_optional(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _theta_from_perturbation_or_compute(
    request: UnifiedPredictionRequestV2,
    payload: NormalizedRemotePayloadV2,
) -> tuple[float, str]:
    """Return (theta_value_k, source_label)."""
    if payload.temperature_perturbation_k is not None:
        return payload.temperature_perturbation_k, "direct_model_prediction"
    if payload.temperature_k is not None:
        # derived from absolute temperature
        return (
            payload.temperature_k - REFERENCE_TEMPERATURE_K,
            "derived_from_temperature",
        )
    # last resort: pure backend-derived theta = T_source - T_ref
    return compute_theta_k(
        source_temperature_k=request.thermal_state.source_temperature_k,
        reference_temperature_k=request.thermal_state.reference_temperature_k,
    ), "derived_from_thermal_state"


def normalize_to_v2(
    request: UnifiedPredictionRequestV2,
    medium: MediumV2,
    derived_geometry: DerivedGeometry2D,
    remote_payload: dict,
    *,
    route: str,
    inference_time_ms: float | None,
) -> dict:
    """Build the final v2 response dict.

    ``remote_payload`` is the raw JSON body the model service returned.
    The function detects its shape, lifts the relevant fields, and
    packages them into the v2 contract.
    """
    payload: NormalizedRemotePayloadV2 = parse_remote_payload(remote_payload)

    # --- thermal -------------------------------------------------------
    temp_k = payload.temperature_k
    theta_k, theta_source = _theta_from_perturbation_or_compute(request, payload)
    if temp_k is None and theta_k is not None:
        temp_k = REFERENCE_TEMPERATURE_K + theta_k

    # --- displacement --------------------------------------------------
    u = payload.displacement_u_m
    v = payload.displacement_v_m
    if u is not None and v is not None:
        magnitude_m: float | None = compute_displacement_magnitude_m(u, v)
        mag_source = "derived_from_u_v"
        components_source = "direct_model_prediction"
    else:
        magnitude_m = payload.max_displacement_m
        mag_source = (
            "derived_from_field_summary"
            if magnitude_m is not None
            else "unavailable"
        )
        components_source = "unavailable"

    # --- diagnostics ---------------------------------------------------
    warnings = list(payload.warnings)
    notes = [DISCLAIMER]
    fallback_used = payload.fallback_used
    fallback_reason = payload.fallback_reason

    response = {
        "schema_version": "2.0",
        "request_id": str(uuid4()),
        "status": "ok",
        "model": {
            "name": request.model.value,
            "version": payload.model_version,
            "route": route,
            "inference_time_ms": _round_optional(inference_time_ms, 2),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
        "material": {
            "id": medium.id,
            "name": medium.name,
            "category": medium.category,
        },
        "geometry": {
            "dimension": request.geometry.dimension,
            "source": request.geometry.source.to_dict(),
            "probe": request.geometry.probe.to_dict(),
            "propagation_vector_m": {
                "dx": derived_geometry.propagation_vector_m[0],
                "dy": derived_geometry.propagation_vector_m[1],
            },
            "unit_direction": {
                "x": derived_geometry.unit_direction[0],
                "y": derived_geometry.unit_direction[1],
            },
            "distance_m": round(derived_geometry.distance_m, 6),
            "azimuth_deg": round(derived_geometry.azimuth_deg, 6),
            "azimuth_convention": derived_geometry.azimuth_convention,
        },
        "prediction": {
            "thermal": {
                "temperature_k": {
                    "value": _round_optional(temp_k, 4),
                    "source": (
                        "direct_model_prediction"
                        if payload.temperature_k is not None
                        else "derived_from_perturbation"
                    ),
                },
                "temperature_perturbation_k": {
                    "value": _round_optional(theta_k, 4),
                    "reference_temperature_k": (
                        request.thermal_state.reference_temperature_k
                    ),
                    "source": theta_source,
                },
            },
            "displacement": {
                "components_m": {
                    "u": _round_optional(u, 12),
                    "v": _round_optional(v, 12),
                },
                "magnitude_m": _round_optional(magnitude_m, 12),
                "components_source": components_source,
                "magnitude_source": mag_source,
            },
            "directional_response": {
                "distance_m": round(derived_geometry.distance_m, 6),
                "azimuth_deg": round(derived_geometry.azimuth_deg, 6),
                "response_magnitude_score": _round_optional(
                    payload.response_magnitude_score, 6
                ),
            },
            "temporal_response": {
                "travel_time_s": _round_optional(payload.travel_time_s, 9),
                "source": (
                    "direct_model_prediction"
                    if payload.travel_time_s is not None
                    else "unavailable"
                ),
            },
        },
        "optional_outputs": {
            "confidence_score": payload.confidence_score,
            "field_summary": {
                "max_displacement_m": _round_optional(
                    payload.max_displacement_m, 9
                ),
                "max_temperature_perturbation_k": _round_optional(
                    payload.max_temperature_perturbation_k, 6
                ),
            },
            "field_grid": payload.field_grid,
            "strain": None,
            "stress": None,
        },
        "diagnostics": {
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "warnings": warnings,
            "notes": notes,
        },
    }
    return response


def normalize_with_geometry(
    request: UnifiedPredictionRequestV2,
    medium: MediumV2,
    remote_payload: dict,
    *,
    route: str,
    inference_time_ms: float | None,
) -> dict:
    """Convenience wrapper: compute derived geometry then normalise."""
    derived = compute_derived_geometry(
        (request.geometry.source.x_m, request.geometry.source.y_m),
        (request.geometry.probe.x_m, request.geometry.probe.y_m),
    )
    return normalize_to_v2(
        request,
        medium,
        DerivedGeometry2D(
            propagation_vector_m=derived.propagation_vector_m,
            distance_m=derived.distance_m,
            unit_direction=derived.unit_direction,
            azimuth_deg=derived.azimuth_deg,
        ),
        remote_payload,
        route=route,
        inference_time_ms=inference_time_ms,
    )
