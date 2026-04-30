from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinn_service.service_schemas import PINNPredictionRequest


@dataclass(frozen=True)
class InferenceFeatures:
    values: np.ndarray
    feature_names: list[str]


LEGACY_FEATURE_NAMES = [
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


def build_feature_vector(
    request: PINNPredictionRequest,
    time_scale: float,
    expected_feature_names: list[str] | None = None,
) -> InferenceFeatures:
    props = request.medium.properties
    rho = float(props.rho)
    vp = _velocity_to_meters_per_second(float(props.vp))
    vs = _velocity_to_meters_per_second(float(props.vs))
    youngs_modulus, poissons_ratio = derive_elastic_parameters(rho, vp, vs)
    source_probe_dx = float(request.probe.x) - float(request.source.x)
    source_probe_dy = float(request.probe.y) - float(request.source.y)
    source_probe_dz = float(request.probe.z) - float(request.source.z)
    source_probe_distance = math.sqrt(source_probe_dx**2 + source_probe_dy**2 + source_probe_dz**2)

    available_features = {
        "x": float(request.probe.x),
        "y": float(request.probe.y),
        "z": float(request.probe.z),
        "t": float(request.scenario.time_ms) * time_scale,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "density": rho,
        "thermal_expansion": float(props.thermal_expansion),
        "thermal_conductivity": float(props.thermal_conductivity),
        "heat_capacity": float(props.heat_capacity),
        "temperature_c": float(request.scenario.temperature_c),
        "pressure_mpa": float(request.scenario.pressure_mpa),
        "source_x": float(request.source.x),
        "source_y": float(request.source.y),
        "source_z": float(request.source.z),
        "source_amplitude": float(request.source.amplitude),
        "source_frequency_hz": float(request.source.frequency_hz),
        "source_dir_x": float(request.source.direction[0]),
        "source_dir_y": float(request.source.direction[1]),
        "source_dir_z": float(request.source.direction[2]),
        "source_probe_dx": source_probe_dx,
        "source_probe_dy": source_probe_dy,
        "source_probe_dz": source_probe_dz,
        "source_probe_distance": source_probe_distance,
        "domain_lx": float(request.domain.size.lx),
        "domain_ly": float(request.domain.size.ly),
        "domain_lz": float(request.domain.size.lz),
        "domain_nx": float(request.domain.resolution.nx),
        "domain_ny": float(request.domain.resolution.ny),
        "domain_nz": float(request.domain.resolution.nz),
    }

    feature_names = expected_feature_names or LEGACY_FEATURE_NAMES
    values = np.asarray([available_features[name] for name in feature_names], dtype=np.float32)
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
    props = request.medium.properties
    temperature_k = float(model_outputs[0])
    base_displacement = np.asarray(model_outputs[1:4], dtype=np.float64)

    source_direction = normalize_vector(np.asarray(request.source.direction, dtype=np.float64), default=[1.0, 0.0, 0.0])
    source_probe_vector = np.asarray(
        [
            float(request.probe.x) - float(request.source.x),
            float(request.probe.y) - float(request.source.y),
            float(request.probe.z) - float(request.source.z),
        ],
        dtype=np.float64,
    )
    source_probe_distance = float(np.linalg.norm(source_probe_vector))
    propagation_direction = normalize_vector(source_probe_vector, default=source_direction)

    domain_is_2d = request.domain.type == "rect_2d"
    vp = _velocity_to_meters_per_second(float(props.vp))
    vs = _velocity_to_meters_per_second(float(props.vs))

    time_ms = float(request.scenario.time_ms)
    amplitude = max(float(request.source.amplitude), 1e-6)
    frequency_hz = max(float(request.source.frequency_hz), 1e-6)
    temperature_c = float(request.scenario.temperature_c)
    pressure_mpa = float(request.scenario.pressure_mpa)
    density = float(props.rho)
    porosity = float(getattr(props, "porosity_effective", 0.0))
    thermal_expansion = float(props.thermal_expansion)
    thermal_conductivity = float(props.thermal_conductivity)
    heat_capacity = float(props.heat_capacity)

    model_direction = normalize_vector(base_displacement, default=propagation_direction)
    model_strength = float(np.linalg.norm(base_displacement))
    response_window = _response_window(time_ms, source_probe_distance, vp, vs, request.source.type)
    attenuation = _distance_attenuation(source_probe_distance, density, porosity)
    thermal_factor = 1.0 + max(temperature_c, 0.0) * thermal_expansion * 18.0
    pressure_factor = 1.0 + pressure_mpa / 2500.0
    frequency_factor = 0.85 + min(frequency_hz / 180.0, 1.1) * 0.35
    source_factor = _source_type_factor(request.source.type)
    compliance_factor = _compliance_factor(vp, vs, density)
    medium_factor = _medium_response_factor(
        density=density,
        porosity=porosity,
        thermal_conductivity=thermal_conductivity,
        thermal_expansion=thermal_expansion,
        vp=vp,
        vs=vs,
    )
    medium_bias = _medium_direction_bias(
        density=density,
        porosity=porosity,
        thermal_conductivity=thermal_conductivity,
        vp=vp,
        vs=vs,
        domain_is_2d=domain_is_2d,
    )

    direction_vector = blend_direction_vectors(
        model_direction=model_direction,
        source_direction=source_direction,
        propagation_direction=propagation_direction,
        response_window=response_window,
        source_factor=source_factor,
        medium_bias=medium_bias,
        domain_is_2d=domain_is_2d,
    )

    azimuth_deg = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
    horizontal = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    elevation_deg = math.degrees(math.atan2(direction_vector[2], horizontal))

    effective_velocity = _effective_velocity(vp, vs, request.source.type, temperature_c, pressure_mpa)
    travel_time_ms = (source_probe_distance / max(effective_velocity, 1e-6)) * 1000.0

    response_strength = (
        amplitude
        * attenuation
        * response_window
        * frequency_factor
        * thermal_factor
        * pressure_factor
        * source_factor
        * compliance_factor
        * medium_factor
    )
    output_signal = max(model_strength, 1e-6)
    displacement_floor = amplitude * attenuation * medium_factor * max(response_window, 0.2) * 4.0e-5
    max_displacement = max(output_signal * response_strength * 25.0, displacement_floor)
    max_temperature_perturbation = (
        abs(temperature_k - reference_temperature_k) * 0.1
        + (abs(temperature_c) / max(heat_capacity, 1.0)) * thermal_conductivity * attenuation * amplitude * medium_factor * 0.9
    )
    magnitude = min(1.0, response_strength * 0.08 + output_signal * 1500.0)

    return {
        "direction_vector": [round(float(component), 6) for component in direction_vector.tolist()],
        "azimuth_deg": round(float(azimuth_deg), 3),
        "elevation_deg": round(float(elevation_deg), 3),
        "magnitude": round(float(magnitude), 6),
        "wave_type": "physics_informed",
        "travel_time_ms": round(float(travel_time_ms), 6),
        "max_displacement": round(max_displacement, 9),
        "max_temperature_perturbation": round(max_temperature_perturbation, 6),
    }


def blend_direction_vectors(
    *,
    model_direction: np.ndarray,
    source_direction: np.ndarray,
    propagation_direction: np.ndarray,
    response_window: float,
    source_factor: float,
    medium_bias: np.ndarray,
    domain_is_2d: bool,
) -> np.ndarray:
    model_weight = 0.3 + response_window * 0.25
    source_weight = 0.2 + source_factor * 0.15
    propagation_weight = 0.55 - source_factor * 0.1

    blended = (
        model_direction * model_weight
        + source_direction * source_weight
        + propagation_direction * propagation_weight
    )
    blended = blended * medium_bias
    if domain_is_2d:
        blended[2] = 0.0
    return normalize_vector(blended, default=propagation_direction if not domain_is_2d else propagation_direction * [1.0, 1.0, 0.0])


def normalize_vector(values: np.ndarray, default: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    magnitude = np.linalg.norm(vector)
    if magnitude < 1e-12:
        fallback = np.asarray(default, dtype=np.float64)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < 1e-12:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback / fallback_norm
    return vector / magnitude


def _response_window(time_ms: float, distance_m: float, vp: float, vs: float, source_type: str) -> float:
    primary_velocity = _effective_velocity(vp, vs, source_type, 0.0, 0.0)
    expected_time_ms = (distance_m / max(primary_velocity, 1e-6)) * 1000.0
    ratio = time_ms / max(expected_time_ms, 1e-6)
    return max(0.15, math.exp(-((ratio - 1.0) ** 2) / 0.7))


def _distance_attenuation(distance_m: float, density: float, porosity: float) -> float:
    density_term = 1.0 + max(density - 2200.0, 0.0) / 8000.0
    porosity_term = 1.0 + porosity * 3.0
    return 1.0 / (1.0 + distance_m * density_term * porosity_term * 1.2)


def _source_type_factor(source_type: str) -> float:
    if source_type == "point_force":
        return 1.15
    if source_type == "coupled_excitation":
        return 1.05
    return 1.0


def _compliance_factor(vp: float, vs: float, density: float) -> float:
    impedance = density * max(vp + vs, 1e-6) * 0.5
    return max(0.35, min(1.4, 5.0e6 / max(impedance, 1e-6)))


def _medium_response_factor(
    *,
    density: float,
    porosity: float,
    thermal_conductivity: float,
    thermal_expansion: float,
    vp: float,
    vs: float,
) -> float:
    stiffness_ratio = vs / max(vp, 1e-6)
    conductivity_term = 0.8 + min(thermal_conductivity / 4.0, 1.2) * 0.45
    porosity_term = 1.0 - min(porosity, 0.35) * 0.75
    density_term = 0.9 + min(density / 3200.0, 1.2) * 0.2
    expansion_term = 0.85 + min(thermal_expansion * 100000.0, 1.4) * 0.12
    stiffness_term = 0.8 + min(stiffness_ratio, 0.75) * 0.5
    return max(0.35, min(1.9, conductivity_term * porosity_term * density_term * expansion_term * stiffness_term))


def _medium_direction_bias(
    *,
    density: float,
    porosity: float,
    thermal_conductivity: float,
    vp: float,
    vs: float,
    domain_is_2d: bool,
) -> np.ndarray:
    x_bias = 1.0 + (thermal_conductivity - 2.2) * 0.06 + (vs / max(vp, 1e-6) - 0.55) * 0.08
    y_bias = 1.0 + (porosity - 0.08) * 0.35 - (density - 2600.0) / 9000.0
    z_bias = 0.0 if domain_is_2d else 1.0 + (density - 2500.0) / 15000.0
    return np.asarray([x_bias, y_bias, z_bias], dtype=np.float64)


def _effective_velocity(vp: float, vs: float, source_type: str, temperature_c: float, pressure_mpa: float) -> float:
    if source_type == "point_force":
        base_velocity = vs
    elif source_type == "coupled_excitation":
        base_velocity = 0.5 * (vp + vs)
    else:
        base_velocity = vp

    thermal_adjustment = 1.0 - max(temperature_c, 0.0) * 0.0002
    pressure_adjustment = 1.0 + pressure_mpa * 0.00008
    return max(250.0, base_velocity * thermal_adjustment * pressure_adjustment)


def _velocity_to_meters_per_second(value: float) -> float:
    if abs(value) < 100.0:
        return value * 1000.0
    return value
