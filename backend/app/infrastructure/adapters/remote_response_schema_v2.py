"""Adapter: parse a remote model-service response into a v2 domain object.

Accepts three observed payload shapes from the four model services:

1. **v2 native** (target): ``prediction_raw`` + ``optional_outputs`` +
   ``diagnostics``. Per api-contract-v2.md §7.

2. **v1 nested** (current FNO): ``prediction`` + ``field_summary`` +
   ``model_version`` + ``diagnostics``.

3. **v1 flat** (current PINN / MGN / Transformer): direction vector,
   azimuth, magnitude, travel_time_ms, max_displacement,
   max_temperature_perturbation at the top level.

The adapter normalises all three into ``NormalizedRemotePayloadV2``.
The downstream normalizer never has to know which shape arrived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedRemotePayloadV2:
    # required physical outputs
    temperature_k: float | None
    temperature_perturbation_k: float | None
    displacement_u_m: float | None
    displacement_v_m: float | None
    travel_time_s: float | None
    # response-magnitude score (comparative, model-specific)
    response_magnitude_score: float | None
    # legacy / optional carryover from v1
    max_displacement_m: float | None
    max_temperature_perturbation_k: float | None
    # v2 optional outputs
    field_grid: dict | None
    confidence_score: float | None
    # provenance / runtime
    model_version: str
    fallback_used: bool
    fallback_reason: str | None
    warnings: list[str] = field(default_factory=list)
    raw_shape: str = "unknown"
    field_summary: dict[str, Any] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)
    available_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _detect_shape(payload: dict[str, Any]) -> str:
    if "prediction_raw" in payload or "schema_version" in payload:
        return "v2"
    if isinstance(payload.get("prediction"), dict) and isinstance(
        payload.get("field_summary"), dict
    ):
        return "v1_nested"
    if "direction_vector" in payload or "travel_time_ms" in payload:
        return "v1_flat"
    return "unknown"


def _parse_v2(payload: dict[str, Any]) -> NormalizedRemotePayloadV2:
    raw = payload.get("prediction_raw") or {}
    disp = raw.get("displacement_m") or {}
    opt = payload.get("optional_outputs") or {}
    fs = opt.get("field_summary") or {}
    diag = payload.get("diagnostics") or {}

    return NormalizedRemotePayloadV2(
        temperature_k=_maybe_float(raw.get("temperature_k")),
        temperature_perturbation_k=_maybe_float(
            raw.get("temperature_perturbation_k")
        ),
        displacement_u_m=_maybe_float(disp.get("u")),
        displacement_v_m=_maybe_float(disp.get("v")),
        travel_time_s=_maybe_float(raw.get("travel_time_s")),
        response_magnitude_score=_maybe_float(raw.get("response_magnitude_score")),
        max_displacement_m=_maybe_float(fs.get("max_displacement_m")),
        max_temperature_perturbation_k=_maybe_float(
            fs.get("max_temperature_perturbation_k")
        ),
        field_grid=opt.get("field_grid"),
        confidence_score=_maybe_float(opt.get("confidence_score")),
        model_version=str(payload.get("model_version", "unknown")),
        fallback_used=bool(diag.get("fallback_used", False)),
        fallback_reason=diag.get("fallback_reason"),
        warnings=list(diag.get("warnings") or []),
        raw_shape="v2",
        field_summary=fs if isinstance(fs, dict) else {},
        field_sources=(
            opt.get("field_sources") if isinstance(opt.get("field_sources"), dict) else {}
        ),
        available_fields=list(opt.get("available_fields") or []),
        missing_fields=list(opt.get("missing_fields") or []),
    )


def _parse_v1_nested(payload: dict[str, Any]) -> NormalizedRemotePayloadV2:
    pred = payload.get("prediction") or {}
    fs = payload.get("field_summary") or {}
    diag = payload.get("diagnostics") or {}

    travel_time_ms = _maybe_float(pred.get("travel_time_ms"))
    return NormalizedRemotePayloadV2(
        temperature_k=None,
        temperature_perturbation_k=_maybe_float(
            fs.get("max_temperature_perturbation")
        ),
        displacement_u_m=None,
        displacement_v_m=None,
        travel_time_s=(travel_time_ms / 1000.0) if travel_time_ms is not None else None,
        response_magnitude_score=_maybe_float(pred.get("magnitude")),
        max_displacement_m=_maybe_float(fs.get("max_displacement")),
        max_temperature_perturbation_k=_maybe_float(
            fs.get("max_temperature_perturbation")
        ),
        field_grid=None,
        confidence_score=None,
        model_version=str(payload.get("model_version", "unknown")),
        fallback_used=bool(diag.get("fallback_used", False)),
        fallback_reason=diag.get("fallback_reason"),
        warnings=list(diag.get("warnings") or []),
        raw_shape="v1_nested",
        field_summary={
            "max_displacement_m": _maybe_float(fs.get("max_displacement")),
            "max_temperature_perturbation_k": _maybe_float(
                fs.get("max_temperature_perturbation")
            ),
        },
    )


def _parse_v1_flat(payload: dict[str, Any]) -> NormalizedRemotePayloadV2:
    diag = payload.get("diagnostics") or {}
    travel_time_ms = _maybe_float(payload.get("travel_time_ms"))
    # PINN exposes per-field outputs in model_outputs
    mo = payload.get("model_outputs") or {}
    names: list[str] = mo.get("feature_names") or []
    values: list[Any] = mo.get("values") or []
    feat = (
        dict(zip(names, values)) if (names and len(values) == len(names)) else {}
    )

    return NormalizedRemotePayloadV2(
        temperature_k=_maybe_float(feat.get("temperature_k")),
        temperature_perturbation_k=_maybe_float(
            payload.get("max_temperature_perturbation")
        ),
        displacement_u_m=_maybe_float(feat.get("disp_x")),
        displacement_v_m=_maybe_float(feat.get("disp_y")),
        travel_time_s=(travel_time_ms / 1000.0) if travel_time_ms is not None else None,
        response_magnitude_score=_maybe_float(payload.get("magnitude")),
        max_displacement_m=_maybe_float(payload.get("max_displacement")),
        max_temperature_perturbation_k=_maybe_float(
            payload.get("max_temperature_perturbation")
        ),
        field_grid=None,
        confidence_score=None,
        model_version=str(payload.get("model_version", "unknown")),
        fallback_used=bool(diag.get("fallback_used", False)),
        fallback_reason=diag.get("fallback_reason"),
        warnings=list(diag.get("warnings") or []),
        raw_shape="v1_flat",
        field_summary={
            "max_displacement_m": _maybe_float(payload.get("max_displacement")),
            "max_temperature_perturbation_k": _maybe_float(
                payload.get("max_temperature_perturbation")
            ),
        },
    )


def parse_remote_payload(payload: dict[str, Any]) -> NormalizedRemotePayloadV2:
    shape = _detect_shape(payload)
    if shape == "v2":
        return _parse_v2(payload)
    if shape == "v1_nested":
        return _parse_v1_nested(payload)
    if shape == "v1_flat":
        return _parse_v1_flat(payload)
    raise ValueError(
        "Unrecognised remote-service payload shape: cannot map to v2 "
        "without 'prediction_raw', 'prediction'/'field_summary', or "
        "'direction_vector'/'travel_time_ms'."
    )
