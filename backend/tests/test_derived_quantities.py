"""Unit tests for backend.app.domain.services.derived_quantities."""
from __future__ import annotations

import math

import pytest

from app.domain.services.derived_quantities import (
    DOMAIN_SIZE_M,
    REFERENCE_TEMPERATURE_K,
    SOURCE_TEMPERATURE_K,
    compute_azimuth_deg,
    compute_derived_geometry,
    compute_displacement_magnitude_m,
    compute_distance_m,
    compute_propagation_vector_m,
    compute_theta_k,
    compute_unit_direction,
    derive_elastic_constants,
    verify_catalog_elastic_consistency,
)


def test_constants_lock_training_invariants():
    assert REFERENCE_TEMPERATURE_K == pytest.approx(273.15)
    assert SOURCE_TEMPERATURE_K == pytest.approx(1500.0)
    assert DOMAIN_SIZE_M == pytest.approx(1.0)


def test_theta_default_is_locked_at_1226_85_k():
    assert compute_theta_k() == pytest.approx(1226.85)


def test_theta_with_overrides():
    assert compute_theta_k(350.0, 293.15) == pytest.approx(56.85)


def test_distance_basic():
    assert compute_distance_m((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert compute_distance_m((0.2, 0.5), (0.8, 0.5)) == pytest.approx(0.6)


def test_propagation_vector_signed_components():
    assert compute_propagation_vector_m((0.2, 0.5), (0.8, 0.5)) == (
        pytest.approx(0.6),
        pytest.approx(0.0),
    )
    assert compute_propagation_vector_m((0.5, 0.5), (0.2, 0.7)) == (
        pytest.approx(-0.3),
        pytest.approx(0.2),
    )


def test_unit_direction_norm_is_one():
    u = compute_unit_direction((0.0, 0.0), (3.0, 4.0))
    assert math.hypot(*u) == pytest.approx(1.0)
    assert u == (pytest.approx(0.6), pytest.approx(0.8))


def test_unit_direction_rejects_coincident_points():
    with pytest.raises(ValueError):
        compute_unit_direction((0.5, 0.5), (0.5, 0.5))


def test_azimuth_atan2_xy_plane():
    # East = 0, North = 90
    assert compute_azimuth_deg((0, 0), (1, 0)) == pytest.approx(0.0)
    assert compute_azimuth_deg((0, 0), (0, 1)) == pytest.approx(90.0)
    assert compute_azimuth_deg((0, 0), (-1, 0)) == pytest.approx(180.0)
    assert compute_azimuth_deg((0, 0), (0, -1)) == pytest.approx(-90.0)


def test_displacement_magnitude():
    assert compute_displacement_magnitude_m(3.0, 4.0) == pytest.approx(5.0)
    assert compute_displacement_magnitude_m(0.0, 0.0) == pytest.approx(0.0)


def test_derived_geometry_packages_everything():
    g = compute_derived_geometry((0.0, 0.0), (3.0, 4.0))
    assert g.propagation_vector_m == (pytest.approx(3.0), pytest.approx(4.0))
    assert g.distance_m == pytest.approx(5.0)
    assert g.unit_direction == (pytest.approx(0.6), pytest.approx(0.8))
    assert g.azimuth_deg == pytest.approx(math.degrees(math.atan2(4.0, 3.0)))


def test_elastic_from_E_nu_matches_known_values():
    # E = 70 GPa, nu = 0.30 -> shear ~26.92 GPa, bulk ~58.33 GPa
    c = derive_elastic_constants(young_modulus_pa=70e9, poisson_ratio=0.30)
    assert c.shear_modulus_pa == pytest.approx(70e9 / (2 * 1.3), rel=1e-9)
    assert c.bulk_modulus_pa == pytest.approx(70e9 / (3 * 0.4), rel=1e-9)
    assert c.lame_lambda_pa == pytest.approx(
        70e9 * 0.30 / (1.3 * 0.4), rel=1e-9
    )


def test_elastic_from_rho_vp_vs_matches_known_values():
    # rho=2725, Vp=5000, Vs=2850 -> shear = rho*Vs^2 = 2.213e10
    c = derive_elastic_constants(
        young_modulus_pa=None,
        poisson_ratio=None,
        rho_kg_m3=2725.0,
        vp_m_s=5000.0,
        vs_m_s=2850.0,
    )
    assert c.shear_modulus_pa == pytest.approx(2725 * 2850 * 2850, rel=1e-12)
    # bulk = rho * (Vp^2 - 4/3 Vs^2)
    expected_bulk = 2725 * (5000**2 - (4 / 3) * 2850**2)
    assert c.bulk_modulus_pa == pytest.approx(expected_bulk, rel=1e-12)
    # lambda = K - 2/3 mu
    assert c.lame_lambda_pa == pytest.approx(
        c.bulk_modulus_pa - (2 / 3) * c.shear_modulus_pa, rel=1e-12
    )


def test_elastic_volumetric_heat_capacity_computed_when_inputs_given():
    c = derive_elastic_constants(
        young_modulus_pa=70e9,
        poisson_ratio=0.25,
        rho_kg_m3=2700.0,
        heat_capacity_j_kgk=1000.0,
    )
    assert c.volumetric_heat_capacity_j_m3k == pytest.approx(2.7e6)


def test_elastic_raises_without_enough_inputs():
    with pytest.raises(ValueError):
        derive_elastic_constants(
            young_modulus_pa=None,
            poisson_ratio=None,
            rho_kg_m3=2700.0,
            vp_m_s=None,
            vs_m_s=None,
        )


def test_elastic_rejects_poisson_half():
    with pytest.raises(ValueError):
        derive_elastic_constants(young_modulus_pa=70e9, poisson_ratio=0.5)


def test_catalog_consistency_passes_within_tolerance():
    # Build E from K, mu through 9Kmu/(3K+mu), then verify round-trip
    K = 5.0e10
    mu = 3.0e10
    E_implied = 9 * K * mu / (3 * K + mu)
    assert verify_catalog_elastic_consistency(E_implied, mu, K)


def test_catalog_consistency_flags_obvious_mismatch():
    K = 5.0e10
    mu = 3.0e10
    E_correct = 9 * K * mu / (3 * K + mu)
    assert not verify_catalog_elastic_consistency(E_correct * 1.5, mu, K)
