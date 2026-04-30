from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from pinn_service.training_data import PRIMARY_OUTPUT_NAMES


def compute_hybrid_pinn_loss(
    *,
    model: torch.nn.Module,
    inputs_scaled: Tensor,
    primary_targets_scaled: Tensor,
    velocity_targets: Tensor,
    input_scaler_mean: Tensor,
    input_scaler_std: Tensor,
    output_scaler_mean: Tensor,
    output_scaler_std: Tensor,
    supervised_weight: float,
    velocity_weight: float,
    thermal_residual_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    inputs_scaled = inputs_scaled.requires_grad_(True)
    predictions_scaled = model(inputs_scaled)

    supervised_loss = F.mse_loss(predictions_scaled, primary_targets_scaled)

    predictions_physical = predictions_scaled * output_scaler_std + output_scaler_mean
    temperature = predictions_physical[:, 0:1]
    disp_x = predictions_physical[:, 1:2]
    disp_y = predictions_physical[:, 2:3]
    disp_z = predictions_physical[:, 3:4]

    grads_t = _gradient(temperature, inputs_scaled)
    grads_ux = _gradient(disp_x, inputs_scaled)
    grads_uy = _gradient(disp_y, inputs_scaled)
    grads_uz = _gradient(disp_z, inputs_scaled)

    x_std = input_scaler_std[0]
    y_std = input_scaler_std[1]
    z_std = input_scaler_std[2]
    t_std = input_scaler_std[3]

    temperature_t = grads_t[:, 3:4] / t_std
    temp_xx = _second_derivative(grads_t[:, 0:1], inputs_scaled, 0) / (x_std * x_std)
    temp_yy = _second_derivative(grads_t[:, 1:2], inputs_scaled, 1) / (y_std * y_std)
    temp_zz = _second_derivative(grads_t[:, 2:3], inputs_scaled, 2) / (z_std * z_std)

    thermal_conductivity = _unscale_feature(inputs_scaled[:, 8:9], input_scaler_mean[8:9], input_scaler_std[8:9])
    density = _unscale_feature(inputs_scaled[:, 6:7], input_scaler_mean[6:7], input_scaler_std[6:7])
    heat_capacity = _unscale_feature(inputs_scaled[:, 9:10], input_scaler_mean[9:10], input_scaler_std[9:10])

    thermal_diffusivity = thermal_conductivity / torch.clamp(density * heat_capacity, min=1e-6)
    thermal_residual = temperature_t - thermal_diffusivity * (temp_xx + temp_yy + temp_zz)
    thermal_residual_loss = torch.mean(thermal_residual.pow(2))

    velocity_prediction = torch.cat(
        [
            grads_ux[:, 3:4] / t_std,
            grads_uy[:, 3:4] / t_std,
            grads_uz[:, 3:4] / t_std,
        ],
        dim=1,
    )
    velocity_consistency_loss = F.mse_loss(velocity_prediction, velocity_targets)

    total_loss = (
        supervised_weight * supervised_loss
        + velocity_weight * velocity_consistency_loss
        + thermal_residual_weight * thermal_residual_loss
    )

    metrics = {
        "supervised_loss": float(supervised_loss.detach().cpu()),
        "velocity_consistency_loss": float(velocity_consistency_loss.detach().cpu()),
        "thermal_residual_loss": float(thermal_residual_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
    return total_loss, metrics


def _gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True,
        create_graph=True,
    )[0]


def _second_derivative(first_derivative_component: Tensor, inputs: Tensor, component_index: int) -> Tensor:
    second_grad = torch.autograd.grad(
        outputs=first_derivative_component,
        inputs=inputs,
        grad_outputs=torch.ones_like(first_derivative_component),
        retain_graph=True,
        create_graph=True,
    )[0]
    return second_grad[:, component_index : component_index + 1]


def _unscale_feature(values_scaled: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return values_scaled * std + mean
