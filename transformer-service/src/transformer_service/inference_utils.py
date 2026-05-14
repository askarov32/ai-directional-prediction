from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from transformer_service.dataset import INPUT_CHANNEL_NAMES, TARGET_CHANNEL_NAMES
from transformer_service.service_schemas import TransformerPredictionRequest


@dataclass(frozen=True)
class InitialStateBuild:
    state: np.ndarray
    coords: np.ndarray


MATERIAL_FIELD_TO_REQUEST: dict[str, str] = {
    "youngs_modulus": "_derived_E",
    "poissons_ratio": "_derived_nu",
    "density": "rho",
    "thermal_expansion": "thermal_expansion",
    "thermal_conductivity": "thermal_conductivity",
    "heat_capacity": "heat_capacity",
}


def derive_elastic_parameters(rho: float, vp: float, vs: float) -> tuple[float, float]:
    mu = rho * vs * vs
    lam = rho * max(vp * vp - 2.0 * vs * vs, 1e-6)
    youngs_modulus = mu * (3.0 * lam + 2.0 * mu) / max(lam + mu, 1e-6)
    poissons_ratio = lam / max(2.0 * (lam + mu), 1e-6)
    return float(youngs_modulus), float(poissons_ratio)


def _velocity_to_mps(value: float) -> float:
    if abs(value) < 100.0:
        return value * 1000.0
    return value


def build_initial_state(
    request: TransformerPredictionRequest,
    coords: np.ndarray,
    reference_temperature_k: float,
) -> InitialStateBuild:
    props = request.medium.properties
    rho = float(props.rho)
    vp = _velocity_to_mps(float(props.vp))
    vs = _velocity_to_mps(float(props.vs))
    youngs_modulus, poissons_ratio = derive_elastic_parameters(rho, vp, vs)

    material_values = {
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "density": rho,
        "thermal_expansion": float(props.thermal_expansion),
        "thermal_conductivity": float(props.thermal_conductivity),
        "heat_capacity": float(props.heat_capacity),
    }
    n_nodes = coords.shape[0]
    n_channels = len(INPUT_CHANNEL_NAMES)
    state = np.zeros((n_nodes, n_channels), dtype=np.float32)
    state[:, 0:3] = coords.astype(np.float32)
    temperature_init = (
        float(request.scenario.temperature_c) + 273.15
        if request.scenario.temperature_c is not None
        else reference_temperature_k
    )
    state[:, INPUT_CHANNEL_NAMES.index("temperature_k")] = temperature_init
    for name, value in material_values.items():
        idx = INPUT_CHANNEL_NAMES.index(name)
        state[:, idx] = value
    return InitialStateBuild(state=state, coords=coords.astype(np.float32))


def normalize_vector(values: np.ndarray, default: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-12:
        fallback = np.asarray(default, dtype=np.float64)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-12:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback / fallback_norm
    return vector / magnitude


def summarize_trajectory(
    trajectory_raw: np.ndarray,
    reference_temperature_k: float,
) -> dict[str, float]:
    """trajectory_raw shape (N, T_steps, F_out=4)."""
    if trajectory_raw.shape[-1] != len(TARGET_CHANNEL_NAMES):
        raise ValueError(
            f"Trajectory channel count {trajectory_raw.shape[-1]} != target channels {len(TARGET_CHANNEL_NAMES)}"
        )
    t_idx = TARGET_CHANNEL_NAMES.index("temperature_k")
    u_idx = TARGET_CHANNEL_NAMES.index("disp_x")
    v_idx = TARGET_CHANNEL_NAMES.index("disp_y")
    w_idx = TARGET_CHANNEL_NAMES.index("disp_z")
    temperature_traj = trajectory_raw[..., t_idx]
    displacement = trajectory_raw[..., [u_idx, v_idx, w_idx]]
    displacement_norm = np.linalg.norm(displacement, axis=-1)
    return {
        "max_displacement": float(np.max(displacement_norm)),
        "max_temperature_perturbation": float(
            np.max(np.abs(temperature_traj - reference_temperature_k))
        ),
        "mean_displacement_vector": displacement.mean(axis=(0, 1)).tolist(),
    }


def build_prediction_payload(
    *,
    request: TransformerPredictionRequest,
    trajectory_raw: np.ndarray,
    reference_temperature_k: float,
) -> dict:
    props = request.medium.properties
    summary = summarize_trajectory(trajectory_raw, reference_temperature_k)

    source_direction = normalize_vector(
        np.asarray(request.source.direction, dtype=np.float64),
        default=np.asarray([1.0, 0.0, 0.0]),
    )
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

    model_direction = normalize_vector(
        np.asarray(summary["mean_displacement_vector"], dtype=np.float64),
        default=propagation_direction,
    )
    blended = (
        0.4 * model_direction
        + 0.4 * propagation_direction
        + 0.2 * source_direction
    )
    domain_is_2d = request.domain.type == "rect_2d"
    if domain_is_2d:
        blended[2] = 0.0
    direction_vector = normalize_vector(blended, default=propagation_direction)

    azimuth_deg = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
    horizontal = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    elevation_deg = math.degrees(math.atan2(direction_vector[2], horizontal))

    vp = _velocity_to_mps(float(props.vp))
    vs = _velocity_to_mps(float(props.vs))
    effective_velocity = max(0.5 * (vp + vs), 1.0)
    travel_time_ms = (source_probe_distance / effective_velocity) * 1000.0

    max_displacement = float(summary["max_displacement"])
    max_temperature_perturbation = float(summary["max_temperature_perturbation"])
    magnitude = float(min(1.0, max_displacement * 1500.0))

    return {
        "direction_vector": [round(float(component), 6) for component in direction_vector.tolist()],
        "azimuth_deg": round(float(azimuth_deg), 3),
        "elevation_deg": round(float(elevation_deg), 3),
        "magnitude": round(magnitude, 6),
        "wave_type": "tokenset",
        "travel_time_ms": round(float(travel_time_ms), 6),
        "max_displacement": round(max_displacement, 9),
        "max_temperature_perturbation": round(max_temperature_perturbation, 6),
    }
