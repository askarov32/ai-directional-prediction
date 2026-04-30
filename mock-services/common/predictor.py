from __future__ import annotations

import math
from typing import Any


SERVICE_OFFSETS = {
    "meshgraphnet": 0.12,
    "fno": -0.08,
}


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return [1.0, 0.0, 0.0]
    return [component / magnitude for component in vector]


def generate_prediction(service_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    scenario = payload.get("scenario", {})
    source = payload.get("source", {})
    probe = payload.get("probe", {})
    medium = payload.get("medium", {})
    properties = medium.get("properties", {})

    offset = SERVICE_OFFSETS.get(service_kind, 0.0)
    dx = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
    dy = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
    dz = float(probe.get("z", 0.0)) - float(source.get("z", 0.0))

    temperature = float(scenario.get("temperature_c", 20.0))
    pressure = float(scenario.get("pressure_mpa", 1.0))
    time_ms = float(scenario.get("time_ms", 1.0))
    vp = float(properties.get("vp", 5.0))

    vector = _normalize(
        [
            dx + 0.25 + offset + temperature / 1500.0,
            dy + 0.1 - offset + pressure / 800.0,
            dz,
        ]
    )

    azimuth_deg = math.degrees(math.atan2(vector[1], vector[0]))
    horizontal = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    elevation_deg = math.degrees(math.atan2(vector[2], horizontal))
    source_probe_distance = math.sqrt(max(dx * dx + dy * dy + dz * dz, 0.0001))

    travel_time_ms = max((source_probe_distance / max(vp, 0.1)) * 10 + time_ms * 0.18 + offset * 5, 0.8)
    max_displacement = 0.0015 + temperature / 250000.0 + pressure / 200000.0 + abs(offset) / 100
    max_temperature_perturbation = 0.7 + temperature / 120.0 + abs(offset) * 2.2

    wave_type = {
        "meshgraphnet": "dominant_p",
        "fno": "coupled_field",
    }.get(service_kind, "dominant_p")

    return {
        "direction_vector": [round(component, 4) for component in vector],
        "azimuth_deg": round(azimuth_deg, 2),
        "elevation_deg": round(elevation_deg, 2),
        "magnitude": 1.0,
        "wave_type": wave_type,
        "travel_time_ms": round(travel_time_ms, 3),
        "max_displacement": round(max_displacement, 6),
        "max_temperature_perturbation": round(max_temperature_perturbation, 6),
        "model_version": f"mock-{service_kind}-v1",
    }
