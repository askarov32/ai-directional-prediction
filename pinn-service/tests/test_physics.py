from __future__ import annotations

import pytest
import torch

from pinn_service.physics import (
    CoordinateScales,
    compute_coupled_thermal_residual,
    compute_lame_parameters,
    compute_strain_tensor,
    compute_stress_tensor,
    compute_temperature_laplacian,
    compute_thermoelastic_gamma,
    compute_wave_residual,
)


def _unit_scales(dtype: torch.dtype = torch.float64) -> CoordinateScales:
    one = torch.tensor(1.0, dtype=dtype)
    return CoordinateScales(x=one, y=one, z=one, t=one)


def _sample_inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [0.2, 0.3, 0.4, 0.5],
            [0.7, 0.1, 0.6, 0.8],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def test_lame_parameters_match_analytic_values_and_preserve_shape() -> None:
    youngs_modulus = torch.full((3, 1), 10.0, dtype=torch.float64)
    poissons_ratio = torch.full((3, 1), 0.25, dtype=torch.float64)

    lambda_, mu = compute_lame_parameters(youngs_modulus, poissons_ratio)

    assert lambda_.shape == youngs_modulus.shape
    assert mu.shape == youngs_modulus.shape
    assert torch.allclose(mu, torch.full_like(mu, 4.0))
    assert torch.allclose(lambda_, torch.full_like(lambda_, 4.0))


def test_lame_parameters_reject_singular_poisson_ratio() -> None:
    with pytest.raises(ValueError):
        compute_lame_parameters(
            torch.ones((1, 1), dtype=torch.float64),
            torch.full((1, 1), 0.5, dtype=torch.float64),
        )


def test_strain_tensor_matches_linear_displacement_field() -> None:
    inputs = _sample_inputs()
    x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]

    strain = compute_strain_tensor(
        disp_x=2.0 * x,
        disp_y=3.0 * y,
        disp_z=4.0 * z,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )

    assert torch.allclose(strain["eps_xx"], torch.full_like(x, 2.0))
    assert torch.allclose(strain["eps_yy"], torch.full_like(y, 3.0))
    assert torch.allclose(strain["eps_zz"], torch.full_like(z, 4.0))
    assert torch.allclose(strain["eps_kk"], torch.full_like(z, 9.0))
    assert torch.allclose(strain["eps_xy"], torch.zeros_like(z))


def test_temperature_laplacian_matches_quadratic_field() -> None:
    inputs = _sample_inputs()
    x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    temperature = x.pow(2) + y.pow(2) + z.pow(2)

    laplacian = compute_temperature_laplacian(
        temperature=temperature,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )

    assert torch.allclose(laplacian, torch.full_like(laplacian, 6.0))


def test_stress_tensor_includes_thermal_diagonal_and_symmetric_shear_terms() -> None:
    values = torch.ones((2, 1), dtype=torch.float64)
    strain = {
        "eps_xx": values * 1.0,
        "eps_yy": values * 2.0,
        "eps_zz": values * 3.0,
        "eps_xy": values * 0.5,
        "eps_xz": values * 0.25,
        "eps_yz": values * 0.75,
        "eps_kk": values * 6.0,
    }

    stress = compute_stress_tensor(
        strain=strain,
        temperature_delta=values * 2.0,
        lambda_=values * 3.0,
        mu=values * 5.0,
        gamma=values * 7.0,
    )

    assert torch.allclose(stress["sigma_xx"], values * 14.0)
    assert torch.allclose(stress["sigma_yy"], values * 24.0)
    assert torch.allclose(stress["sigma_zz"], values * 34.0)
    assert torch.allclose(stress["sigma_xy"], values * 5.0)
    assert torch.allclose(stress["sigma_xz"], values * 2.5)
    assert torch.allclose(stress["sigma_yz"], values * 7.5)


def test_wave_residual_shape_scalar_loss_and_backward() -> None:
    inputs = _sample_inputs()
    x, y, z, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4]
    disp_x = x.pow(2) + t.pow(2)
    disp_y = y.pow(2) + t.pow(2)
    disp_z = z.pow(2) + t.pow(2)
    temperature = x + y + z + t
    density = torch.full_like(x, 2.0)
    lambda_ = torch.full_like(x, 3.0)
    mu = torch.full_like(x, 5.0)
    gamma = torch.full_like(x, 0.25)

    strain = compute_strain_tensor(
        disp_x=disp_x,
        disp_y=disp_y,
        disp_z=disp_z,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )
    stress = compute_stress_tensor(
        strain=strain,
        temperature_delta=temperature - 0.5,
        lambda_=lambda_,
        mu=mu,
        gamma=gamma,
    )
    residual = compute_wave_residual(
        disp_x=disp_x,
        disp_y=disp_y,
        disp_z=disp_z,
        density=density,
        stress=stress,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )
    loss = torch.mean(torch.sum(residual.pow(2), dim=1, keepdim=True))

    assert residual.shape == (2, 3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))


def test_coupled_thermal_residual_shape_scalar_loss_and_backward() -> None:
    inputs = _sample_inputs()
    x, y, z, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4]
    temperature = x.pow(2) + y.pow(2) + z.pow(2) + t
    disp_x = x * t
    disp_y = y * t
    disp_z = z * t
    density = torch.full_like(x, 2.0)
    heat_capacity = torch.full_like(x, 3.0)
    thermal_conductivity = torch.full_like(x, 4.0)
    lambda_, mu = compute_lame_parameters(torch.full_like(x, 10.0), torch.full_like(x, 0.25))
    gamma = compute_thermoelastic_gamma(lambda_, mu, torch.full_like(x, 0.01))
    strain = compute_strain_tensor(
        disp_x=disp_x,
        disp_y=disp_y,
        disp_z=disp_z,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )

    residual = compute_coupled_thermal_residual(
        temperature=temperature,
        strain=strain,
        density=density,
        heat_capacity=heat_capacity,
        thermal_conductivity=thermal_conductivity,
        gamma=gamma,
        reference_temperature_k=293.15,
        inputs_scaled=inputs,
        scales=_unit_scales(),
    )
    loss = torch.mean(residual.pow(2))

    assert residual.shape == (2, 1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))
