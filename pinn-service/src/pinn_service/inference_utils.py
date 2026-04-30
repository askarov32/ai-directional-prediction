from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinn_service.service_schemas import PINNPredictionRequest


@dataclass(frozen=True)
class InferenceFeatures:
    values: np.ndarray
    feature_names: list[str]


def build_feature_vector(request: PINNPredictionRequest, time_scale: float) -> InferenceFeatures:
    props = request.medium.properties
    rho = float(props.rho)
    vp = _velocity_to_meters_per_second(float(props.vp))
    vs = _velocity_to_meters_per_second(float(props.vs))
    youngs_modulus, poissons_ratio = derive_elastic_parameters(rho, vp, vs)

    feature_names = [
        "x",
        "y",
        "z",
        "t",
        "youngs_modulus",
        "poissons_ratio",
        "density",
        "thermal_expansion",
        "thermal_conductivity",
        "heat_capacity",
    ]
    values = np.asarray(
        [
            float(request.probe.x),
            float(request.probe.y),
            float(request.probe.z),
            float(request.scenario.time_ms) * time_scale,
            youngs_modulus,
            poissons_ratio,
            rho,
            float(props.thermal_expansion),
            float(props.thermal_conductivity),
            float(props.heat_capacity),
        ],
        dtype=np.float32,
    )
    return InferenceFeatures(values=values, feature_names=feature_names)


def derive_elastic_parameters(rho: float, vp: float, vs: float) -> tuple[float, float]:
    mu = rho * vs * vs
    lam = rho * max(vp * vp - 2.0 * vs * vs, 1e-6)
    youngs_modulus = mu * (3.0 * lam + 2.0 * mu) / max(lam + mu, 1e-6)
    poissons_ratio = lam / max(2.0 * (lam + mu), 1e-6)
    return float(youngs_modulus), float(poissons_ratio)


def build_prediction_payload(
    *,
    request: PINNPredictionRequest,
    model_outputs: np.ndarray,
    reference_temperature_k: float,
) -> dict:
    temperature_k = float(model_outputs[0])
    displacement_vector = np.asarray(model_outputs[1:4], dtype=np.float64)
    direction_vector = normalize_direction(displacement_vector, np.asarray(request.source.direction, dtype=np.float64))

    azimuth_deg = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
    horizontal = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    elevation_deg = math.degrees(math.atan2(direction_vector[2], horizontal))

    max_displacement = float(np.linalg.norm(displacement_vector))
    source_probe_distance = math.sqrt(
        (float(request.probe.x) - float(request.source.x)) ** 2
        + (float(request.probe.y) - float(request.source.y)) ** 2
        + (float(request.probe.z) - float(request.source.z)) ** 2
    )
    vp = _velocity_to_meters_per_second(float(request.medium.properties.vp))
    travel_time_ms = (source_probe_distance / max(vp, 1e-6)) * 1000.0

    return {
        "direction_vector": [round(float(component), 6) for component in direction_vector.tolist()],
        "azimuth_deg": round(float(azimuth_deg), 3),
        "elevation_deg": round(float(elevation_deg), 3),
        "magnitude": round(float(np.linalg.norm(direction_vector)), 6),
        "wave_type": "physics_informed",
        "travel_time_ms": round(float(travel_time_ms), 6),
        "max_displacement": round(max_displacement, 9),
        "max_temperature_perturbation": round(abs(temperature_k - reference_temperature_k), 6),
    }


def normalize_direction(displacement_vector: np.ndarray, source_direction: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(displacement_vector)
    if magnitude < 1e-12:
        fallback = np.asarray(source_direction, dtype=np.float64)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < 1e-12:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback / fallback_norm
    return displacement_vector / magnitude


def _velocity_to_meters_per_second(value: float) -> float:
    if abs(value) < 100.0:
        return value * 1000.0
    return value
