from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CoordinateScales:
    x: Tensor
    y: Tensor
    z: Tensor
    t: Tensor


def compute_lame_parameters(youngs_modulus: Tensor, poissons_ratio: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    if torch.any(poissons_ratio <= -1.0 + eps) or torch.any(torch.abs(1.0 - 2.0 * poissons_ratio) <= eps):
        raise ValueError("Poisson ratio must be greater than -1 and safely different from 0.5.")

    mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lambda_ = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    return lambda_, mu


def compute_thermoelastic_gamma(lambda_: Tensor, mu: Tensor, thermal_expansion: Tensor) -> Tensor:
    return (3.0 * lambda_ + 2.0 * mu) * thermal_expansion


def gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    if not outputs.requires_grad:
        return torch.zeros_like(inputs)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(inputs)
    return grad


def first_derivative(outputs: Tensor, inputs: Tensor, component_index: int, scale: Tensor | float) -> Tensor:
    return gradient(outputs, inputs)[:, component_index : component_index + 1] / scale


def second_derivative(outputs: Tensor, inputs: Tensor, component_index: int, scale: Tensor | float) -> Tensor:
    first_scaled = gradient(outputs, inputs)[:, component_index : component_index + 1]
    second_scaled = gradient(first_scaled, inputs)[:, component_index : component_index + 1]
    return second_scaled / (scale * scale)


def compute_strain_tensor(
    *,
    disp_x: Tensor,
    disp_y: Tensor,
    disp_z: Tensor,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> dict[str, Tensor]:
    u_x = first_derivative(disp_x, inputs_scaled, 0, scales.x)
    u_y = first_derivative(disp_x, inputs_scaled, 1, scales.y)
    u_z = first_derivative(disp_x, inputs_scaled, 2, scales.z)
    v_x = first_derivative(disp_y, inputs_scaled, 0, scales.x)
    v_y = first_derivative(disp_y, inputs_scaled, 1, scales.y)
    v_z = first_derivative(disp_y, inputs_scaled, 2, scales.z)
    w_x = first_derivative(disp_z, inputs_scaled, 0, scales.x)
    w_y = first_derivative(disp_z, inputs_scaled, 1, scales.y)
    w_z = first_derivative(disp_z, inputs_scaled, 2, scales.z)

    eps_xx = u_x
    eps_yy = v_y
    eps_zz = w_z
    eps_xy = 0.5 * (u_y + v_x)
    eps_xz = 0.5 * (u_z + w_x)
    eps_yz = 0.5 * (v_z + w_y)
    eps_kk = eps_xx + eps_yy + eps_zz

    return {
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_zz": eps_zz,
        "eps_xy": eps_xy,
        "eps_xz": eps_xz,
        "eps_yz": eps_yz,
        "eps_kk": eps_kk,
    }


def compute_plane_strain_tensor(
    *,
    disp_x: Tensor,
    disp_y: Tensor,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> dict[str, Tensor]:
    u_x = first_derivative(disp_x, inputs_scaled, 0, scales.x)
    u_y = first_derivative(disp_x, inputs_scaled, 1, scales.y)
    v_x = first_derivative(disp_y, inputs_scaled, 0, scales.x)
    v_y = first_derivative(disp_y, inputs_scaled, 1, scales.y)

    zeros = torch.zeros_like(u_x)
    eps_xy = 0.5 * (u_y + v_x)
    eps_kk = u_x + v_y

    return {
        "eps_xx": u_x,
        "eps_yy": v_y,
        "eps_zz": zeros,
        "eps_xy": eps_xy,
        "eps_xz": zeros,
        "eps_yz": zeros,
        "eps_kk": eps_kk,
    }


def compute_stress_tensor(
    *,
    strain: dict[str, Tensor],
    temperature_delta: Tensor,
    lambda_: Tensor,
    mu: Tensor,
    gamma: Tensor,
) -> dict[str, Tensor]:
    eps_kk = strain["eps_kk"]
    sigma_xx = lambda_ * eps_kk + 2.0 * mu * strain["eps_xx"] - gamma * temperature_delta
    sigma_yy = lambda_ * eps_kk + 2.0 * mu * strain["eps_yy"] - gamma * temperature_delta
    sigma_zz = lambda_ * eps_kk + 2.0 * mu * strain["eps_zz"] - gamma * temperature_delta
    sigma_xy = 2.0 * mu * strain["eps_xy"]
    sigma_xz = 2.0 * mu * strain["eps_xz"]
    sigma_yz = 2.0 * mu * strain["eps_yz"]

    return {
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_zz": sigma_zz,
        "sigma_xy": sigma_xy,
        "sigma_xz": sigma_xz,
        "sigma_yz": sigma_yz,
    }


def compute_stress_divergence(
    *,
    stress: dict[str, Tensor],
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    div_x = (
        first_derivative(stress["sigma_xx"], inputs_scaled, 0, scales.x)
        + first_derivative(stress["sigma_xy"], inputs_scaled, 1, scales.y)
        + first_derivative(stress["sigma_xz"], inputs_scaled, 2, scales.z)
    )
    div_y = (
        first_derivative(stress["sigma_xy"], inputs_scaled, 0, scales.x)
        + first_derivative(stress["sigma_yy"], inputs_scaled, 1, scales.y)
        + first_derivative(stress["sigma_yz"], inputs_scaled, 2, scales.z)
    )
    div_z = (
        first_derivative(stress["sigma_xz"], inputs_scaled, 0, scales.x)
        + first_derivative(stress["sigma_yz"], inputs_scaled, 1, scales.y)
        + first_derivative(stress["sigma_zz"], inputs_scaled, 2, scales.z)
    )
    return torch.cat([div_x, div_y, div_z], dim=1)


def compute_stress_divergence_2d(
    *,
    stress: dict[str, Tensor],
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    div_x = first_derivative(stress["sigma_xx"], inputs_scaled, 0, scales.x) + first_derivative(
        stress["sigma_xy"],
        inputs_scaled,
        1,
        scales.y,
    )
    div_y = first_derivative(stress["sigma_xy"], inputs_scaled, 0, scales.x) + first_derivative(
        stress["sigma_yy"],
        inputs_scaled,
        1,
        scales.y,
    )
    return torch.cat([div_x, div_y], dim=1)


def compute_wave_residual(
    *,
    disp_x: Tensor,
    disp_y: Tensor,
    disp_z: Tensor,
    density: Tensor,
    stress: dict[str, Tensor],
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    acceleration = torch.cat(
        [
            second_derivative(disp_x, inputs_scaled, 3, scales.t),
            second_derivative(disp_y, inputs_scaled, 3, scales.t),
            second_derivative(disp_z, inputs_scaled, 3, scales.t),
        ],
        dim=1,
    )
    return density * acceleration - compute_stress_divergence(stress=stress, inputs_scaled=inputs_scaled, scales=scales)


def compute_wave_residual_2d(
    *,
    disp_x: Tensor,
    disp_y: Tensor,
    density: Tensor,
    stress: dict[str, Tensor],
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    acceleration = torch.cat(
        [
            second_derivative(disp_x, inputs_scaled, 3, scales.t),
            second_derivative(disp_y, inputs_scaled, 3, scales.t),
        ],
        dim=1,
    )
    return density * acceleration - compute_stress_divergence_2d(stress=stress, inputs_scaled=inputs_scaled, scales=scales)


def compute_temperature_laplacian(*, temperature: Tensor, inputs_scaled: Tensor, scales: CoordinateScales) -> Tensor:
    return (
        second_derivative(temperature, inputs_scaled, 0, scales.x)
        + second_derivative(temperature, inputs_scaled, 1, scales.y)
        + second_derivative(temperature, inputs_scaled, 2, scales.z)
    )


def compute_temperature_laplacian_2d(*, temperature: Tensor, inputs_scaled: Tensor, scales: CoordinateScales) -> Tensor:
    return second_derivative(temperature, inputs_scaled, 0, scales.x) + second_derivative(
        temperature,
        inputs_scaled,
        1,
        scales.y,
    )


def compute_simple_heat_residual(
    *,
    temperature: Tensor,
    density: Tensor,
    heat_capacity: Tensor,
    thermal_conductivity: Tensor,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    temperature_t = first_derivative(temperature, inputs_scaled, 3, scales.t)
    thermal_diffusivity = thermal_conductivity / torch.clamp(density * heat_capacity, min=1e-12)
    return temperature_t - thermal_diffusivity * compute_temperature_laplacian(
        temperature=temperature,
        inputs_scaled=inputs_scaled,
        scales=scales,
    )


def compute_simple_heat_residual_2d(
    *,
    temperature: Tensor,
    density: Tensor,
    heat_capacity: Tensor,
    thermal_conductivity: Tensor,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    temperature_t = first_derivative(temperature, inputs_scaled, 3, scales.t)
    thermal_diffusivity = thermal_conductivity / torch.clamp(density * heat_capacity, min=1e-12)
    return temperature_t - thermal_diffusivity * compute_temperature_laplacian_2d(
        temperature=temperature,
        inputs_scaled=inputs_scaled,
        scales=scales,
    )


def compute_coupled_thermal_residual(
    *,
    temperature: Tensor,
    strain: dict[str, Tensor],
    density: Tensor,
    heat_capacity: Tensor,
    thermal_conductivity: Tensor,
    gamma: Tensor,
    reference_temperature_k: float,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    temperature_t = first_derivative(temperature, inputs_scaled, 3, scales.t)
    laplacian_t = compute_temperature_laplacian(temperature=temperature, inputs_scaled=inputs_scaled, scales=scales)
    eps_kk_t = first_derivative(strain["eps_kk"], inputs_scaled, 3, scales.t)
    return density * heat_capacity * temperature_t - thermal_conductivity * laplacian_t + gamma * reference_temperature_k * eps_kk_t


def compute_coupled_thermal_residual_2d(
    *,
    temperature: Tensor,
    strain: dict[str, Tensor],
    density: Tensor,
    heat_capacity: Tensor,
    thermal_conductivity: Tensor,
    gamma: Tensor,
    reference_temperature_k: float,
    inputs_scaled: Tensor,
    scales: CoordinateScales,
) -> Tensor:
    temperature_t = first_derivative(temperature, inputs_scaled, 3, scales.t)
    laplacian_t = compute_temperature_laplacian_2d(temperature=temperature, inputs_scaled=inputs_scaled, scales=scales)
    eps_kk_t = first_derivative(strain["eps_kk"], inputs_scaled, 3, scales.t)
    return density * heat_capacity * temperature_t - thermal_conductivity * laplacian_t + gamma * reference_temperature_k * eps_kk_t
