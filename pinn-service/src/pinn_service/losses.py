from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from pinn_service.physics import (
    CoordinateScales,
    compute_coupled_thermal_residual,
    compute_coupled_thermal_residual_2d,
    compute_lame_parameters,
    compute_plane_strain_tensor,
    compute_simple_heat_residual,
    compute_simple_heat_residual_2d,
    compute_strain_tensor,
    compute_stress_tensor,
    compute_thermoelastic_gamma,
    compute_wave_residual,
    compute_wave_residual_2d,
    first_derivative,
)


INPUT_FEATURE_INDEXES = {
    "x": 0,
    "y": 1,
    "z": 2,
    "t": 3,
    "youngs_modulus": 4,
    "poissons_ratio": 5,
    "density": 6,
    "thermal_expansion": 7,
    "thermal_conductivity": 8,
    "heat_capacity": 9,
}


PhysicsMode = Literal["coupled_thermoelastic", "simple_heat", "plane_strain_2d"]
LossBalanceMode = Literal["fixed", "normalize"]


def compute_hybrid_pinn_loss(
    *,
    model: torch.nn.Module,
    inputs_scaled: Tensor,
    primary_targets_scaled: Tensor,
    velocity_targets: Tensor | None,
    input_scaler_mean: Tensor,
    input_scaler_std: Tensor,
    output_scaler_mean: Tensor,
    output_scaler_std: Tensor,
    supervised_weight: float,
    velocity_weight: float,
    wave_residual_weight: float,
    thermal_residual_weight: float,
    reference_temperature_k: float,
    physics_mode: PhysicsMode = "coupled_thermoelastic",
    loss_balance_mode: LossBalanceMode = "fixed",
    supervised_loss_scale: float = 1.0,
    velocity_loss_scale: float = 1.0,
    wave_residual_loss_scale: float = 1.0,
    thermal_residual_loss_scale: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    _validate_training_shapes(inputs_scaled, primary_targets_scaled, input_scaler_mean, input_scaler_std)

    inputs_scaled = inputs_scaled.requires_grad_(True)
    predictions_scaled = model(inputs_scaled)

    supervised_loss = F.mse_loss(predictions_scaled, primary_targets_scaled)
    predictions_physical = predictions_scaled * output_scaler_std + output_scaler_mean

    temperature = predictions_physical[:, 0:1]
    disp_x = predictions_physical[:, 1:2]
    disp_y = predictions_physical[:, 2:3]
    disp_z = predictions_physical[:, 3:4]

    scales = CoordinateScales(
        x=input_scaler_std[INPUT_FEATURE_INDEXES["x"]],
        y=input_scaler_std[INPUT_FEATURE_INDEXES["y"]],
        z=input_scaler_std[INPUT_FEATURE_INDEXES["z"]],
        t=input_scaler_std[INPUT_FEATURE_INDEXES["t"]],
    )

    material = _unscale_material_features(inputs_scaled, input_scaler_mean, input_scaler_std)
    lambda_, mu = compute_lame_parameters(material["youngs_modulus"], material["poissons_ratio"])
    gamma = compute_thermoelastic_gamma(lambda_, mu, material["thermal_expansion"])

    is_plane_strain_2d = physics_mode == "plane_strain_2d"
    if is_plane_strain_2d:
        strain = compute_plane_strain_tensor(
            disp_x=disp_x,
            disp_y=disp_y,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    else:
        strain = compute_strain_tensor(
            disp_x=disp_x,
            disp_y=disp_y,
            disp_z=disp_z,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    stress = compute_stress_tensor(
        strain=strain,
        temperature_delta=temperature - reference_temperature_k,
        lambda_=lambda_,
        mu=mu,
        gamma=gamma,
    )

    if is_plane_strain_2d:
        wave_residual = compute_wave_residual_2d(
            disp_x=disp_x,
            disp_y=disp_y,
            density=material["density"],
            stress=stress,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    else:
        wave_residual = compute_wave_residual(
            disp_x=disp_x,
            disp_y=disp_y,
            disp_z=disp_z,
            density=material["density"],
            stress=stress,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    wave_residual_loss = torch.mean(torch.sum(wave_residual.pow(2), dim=1, keepdim=True))

    if physics_mode == "coupled_thermoelastic":
        thermal_residual = compute_coupled_thermal_residual(
            temperature=temperature,
            strain=strain,
            density=material["density"],
            heat_capacity=material["heat_capacity"],
            thermal_conductivity=material["thermal_conductivity"],
            gamma=gamma,
            reference_temperature_k=reference_temperature_k,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    elif physics_mode == "plane_strain_2d":
        thermal_residual = compute_coupled_thermal_residual_2d(
            temperature=temperature,
            strain=strain,
            density=material["density"],
            heat_capacity=material["heat_capacity"],
            thermal_conductivity=material["thermal_conductivity"],
            gamma=gamma,
            reference_temperature_k=reference_temperature_k,
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
    elif physics_mode == "simple_heat":
        thermal_residual = compute_simple_heat_residual(
            temperature=temperature,
            density=material["density"],
            heat_capacity=material["heat_capacity"],
            thermal_conductivity=material["thermal_conductivity"],
            inputs_scaled=inputs_scaled,
            scales=scales,
        )
        wave_residual_loss = torch.zeros((), dtype=supervised_loss.dtype, device=supervised_loss.device)
    else:
        raise ValueError(f"Unsupported physics_mode: {physics_mode}")

    thermal_residual_loss = torch.mean(thermal_residual.pow(2))
    velocity_consistency_loss = _velocity_consistency_loss(
        disp_x=disp_x,
        disp_y=disp_y,
        disp_z=disp_z,
        inputs_scaled=inputs_scaled,
        time_scale=scales.t,
        velocity_targets=velocity_targets,
        planar=is_plane_strain_2d,
    )

    normalized_losses = _normalize_loss_components(
        supervised_loss=supervised_loss,
        velocity_consistency_loss=velocity_consistency_loss,
        wave_residual_loss=wave_residual_loss,
        thermal_residual_loss=thermal_residual_loss,
        loss_balance_mode=loss_balance_mode,
        supervised_loss_scale=supervised_loss_scale,
        velocity_loss_scale=velocity_loss_scale,
        wave_residual_loss_scale=wave_residual_loss_scale,
        thermal_residual_loss_scale=thermal_residual_loss_scale,
    )

    total_loss = (
        supervised_weight * normalized_losses["supervised_loss"]
        + velocity_weight * normalized_losses["velocity_consistency_loss"]
        + wave_residual_weight * normalized_losses["wave_residual_loss"]
        + thermal_residual_weight * normalized_losses["thermal_residual_loss"]
    )

    metrics = {
        "supervised_loss": float(supervised_loss.detach().cpu()),
        "velocity_consistency_loss": float(velocity_consistency_loss.detach().cpu()),
        "wave_residual_loss": float(wave_residual_loss.detach().cpu()),
        "thermal_residual_loss": float(thermal_residual_loss.detach().cpu()),
        "normalized_supervised_loss": float(normalized_losses["supervised_loss"].detach().cpu()),
        "normalized_velocity_consistency_loss": float(normalized_losses["velocity_consistency_loss"].detach().cpu()),
        "normalized_wave_residual_loss": float(normalized_losses["wave_residual_loss"].detach().cpu()),
        "normalized_thermal_residual_loss": float(normalized_losses["thermal_residual_loss"].detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
    return total_loss, metrics


def _velocity_consistency_loss(
    *,
    disp_x: Tensor,
    disp_y: Tensor,
    disp_z: Tensor,
    inputs_scaled: Tensor,
    time_scale: Tensor,
    velocity_targets: Tensor | None,
    planar: bool = False,
) -> Tensor:
    velocity_components = [
        first_derivative(disp_x, inputs_scaled, 3, time_scale),
        first_derivative(disp_y, inputs_scaled, 3, time_scale),
    ]
    if not planar:
        velocity_components.append(first_derivative(disp_z, inputs_scaled, 3, time_scale))
    velocity_prediction = torch.cat(velocity_components, dim=1)
    if velocity_targets is None:
        return torch.zeros((), dtype=velocity_prediction.dtype, device=velocity_prediction.device)
    if planar:
        velocity_targets = velocity_targets[:, :2]
    return F.mse_loss(velocity_prediction, velocity_targets)


def _unscale_material_features(inputs_scaled: Tensor, input_scaler_mean: Tensor, input_scaler_std: Tensor) -> dict[str, Tensor]:
    return {
        name: inputs_scaled[:, index : index + 1] * input_scaler_std[index] + input_scaler_mean[index]
        for name, index in INPUT_FEATURE_INDEXES.items()
        if name not in {"x", "y", "z", "t"}
    }


def _validate_training_shapes(
    inputs_scaled: Tensor,
    primary_targets_scaled: Tensor,
    input_scaler_mean: Tensor,
    input_scaler_std: Tensor,
) -> None:
    if inputs_scaled.ndim != 2 or inputs_scaled.shape[1] < len(INPUT_FEATURE_INDEXES):
        raise ValueError("inputs_scaled must be a 2D tensor containing the 10 PINN input features.")
    if primary_targets_scaled.ndim != 2 or primary_targets_scaled.shape[1] != 4:
        raise ValueError("primary_targets_scaled must contain [T, u, v, w].")
    if input_scaler_mean.numel() < len(INPUT_FEATURE_INDEXES) or input_scaler_std.numel() < len(INPUT_FEATURE_INDEXES):
        raise ValueError("input scaler statistics must contain all 10 PINN input features.")


def _normalize_loss_components(
    *,
    supervised_loss: Tensor,
    velocity_consistency_loss: Tensor,
    wave_residual_loss: Tensor,
    thermal_residual_loss: Tensor,
    loss_balance_mode: LossBalanceMode,
    supervised_loss_scale: float,
    velocity_loss_scale: float,
    wave_residual_loss_scale: float,
    thermal_residual_loss_scale: float,
) -> dict[str, Tensor]:
    components = {
        "supervised_loss": supervised_loss,
        "velocity_consistency_loss": velocity_consistency_loss,
        "wave_residual_loss": wave_residual_loss,
        "thermal_residual_loss": thermal_residual_loss,
    }
    if loss_balance_mode == "fixed":
        return components
    if loss_balance_mode != "normalize":
        raise ValueError(f"Unsupported loss_balance_mode: {loss_balance_mode}")

    return {
        "supervised_loss": supervised_loss / _safe_scale(supervised_loss_scale),
        "velocity_consistency_loss": velocity_consistency_loss / _safe_scale(velocity_loss_scale),
        "wave_residual_loss": wave_residual_loss / _safe_scale(wave_residual_loss_scale),
        "thermal_residual_loss": thermal_residual_loss / _safe_scale(thermal_residual_loss_scale),
    }


def _safe_scale(value: float) -> float:
    return value if value > 1e-12 else 1.0
