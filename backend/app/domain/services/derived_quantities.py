"""Pure functions that compute every derived quantity of the v2 contract.

All formulas are listed in PDF §6.2 of api_contract_redesign_plan.pdf and
in §4.2/§3.2 of docs/api-contract-v2.md. Keeping them in one module
makes the contract single-source-of-truth and easy to unit-test.

No FastAPI / Pydantic / IO here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


REFERENCE_TEMPERATURE_K = 273.15
"""Rock initial temperature (locked at 0 °C across the v2 prototype)."""

SOURCE_TEMPERATURE_K = 1500.0
"""Source amplitude in absolute temperature (locked across the v2 prototype)."""

DOMAIN_SIZE_M = 1.0
"""Domain side length (locked at 1 m x 1 m)."""


@dataclass(frozen=True)
class DerivedGeometry:
    propagation_vector_m: tuple[float, float]
    distance_m: float
    unit_direction: tuple[float, float]
    azimuth_deg: float


@dataclass(frozen=True)
class DerivedElasticConstants:
    shear_modulus_pa: float
    bulk_modulus_pa: float
    lame_lambda_pa: float
    volumetric_heat_capacity_j_m3k: float | None


def compute_theta_k(
    source_temperature_k: float = SOURCE_TEMPERATURE_K,
    reference_temperature_k: float = REFERENCE_TEMPERATURE_K,
) -> float:
    """Temperature perturbation theta = T_source - T_ref (kelvin)."""
    return float(source_temperature_k) - float(reference_temperature_k)


def compute_distance_m(
    source: tuple[float, float], probe: tuple[float, float]
) -> float:
    dx = probe[0] - source[0]
    dy = probe[1] - source[1]
    return math.hypot(dx, dy)


def compute_propagation_vector_m(
    source: tuple[float, float], probe: tuple[float, float]
) -> tuple[float, float]:
    return (probe[0] - source[0], probe[1] - source[1])


def compute_unit_direction(
    source: tuple[float, float], probe: tuple[float, float]
) -> tuple[float, float]:
    distance = compute_distance_m(source, probe)
    if distance == 0:
        raise ValueError("source and probe coincide; unit direction is undefined.")
    dx, dy = compute_propagation_vector_m(source, probe)
    return (dx / distance, dy / distance)


def compute_azimuth_deg(
    source: tuple[float, float], probe: tuple[float, float]
) -> float:
    """Propagation azimuth in degrees, atan2 convention, xy-plane."""
    dx, dy = compute_propagation_vector_m(source, probe)
    return math.degrees(math.atan2(dy, dx))


def compute_displacement_magnitude_m(u_m: float, v_m: float) -> float:
    return math.hypot(u_m, v_m)


def compute_derived_geometry(
    source: tuple[float, float], probe: tuple[float, float]
) -> DerivedGeometry:
    vec = compute_propagation_vector_m(source, probe)
    distance = compute_distance_m(source, probe)
    if distance == 0:
        raise ValueError("source and probe coincide; derived geometry is undefined.")
    return DerivedGeometry(
        propagation_vector_m=vec,
        distance_m=distance,
        unit_direction=(vec[0] / distance, vec[1] / distance),
        azimuth_deg=math.degrees(math.atan2(vec[1], vec[0])),
    )


def derive_elastic_constants(
    young_modulus_pa: float | None,
    poisson_ratio: float | None,
    rho_kg_m3: float | None = None,
    vp_m_s: float | None = None,
    vs_m_s: float | None = None,
    heat_capacity_j_kgk: float | None = None,
) -> DerivedElasticConstants:
    """Derive shear, bulk, Lame parameters.

    Prefers (E, nu) when available; falls back to (rho, Vp, Vs) under
    isotropic-elasticity assumptions. Raises ``ValueError`` if neither
    pathway is feasible.
    """
    if young_modulus_pa is not None and poisson_ratio is not None:
        E = float(young_modulus_pa)
        nu = float(poisson_ratio)
        if nu == 0.5:
            raise ValueError("Poisson ratio nu = 0.5 is degenerate (incompressible).")
        shear = E / (2.0 * (1.0 + nu))
        bulk = E / (3.0 * (1.0 - 2.0 * nu))
        lame_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    elif rho_kg_m3 is not None and vp_m_s is not None and vs_m_s is not None:
        rho = float(rho_kg_m3)
        vp = float(vp_m_s)
        vs = float(vs_m_s)
        shear = rho * vs * vs
        bulk = rho * (vp * vp - (4.0 / 3.0) * vs * vs)
        lame_lambda = bulk - (2.0 / 3.0) * shear
    else:
        raise ValueError(
            "Cannot derive elastic constants: provide either (E, nu) "
            "or (rho, Vp, Vs)."
        )

    volumetric_C: float | None = None
    if rho_kg_m3 is not None and heat_capacity_j_kgk is not None:
        volumetric_C = float(rho_kg_m3) * float(heat_capacity_j_kgk)

    return DerivedElasticConstants(
        shear_modulus_pa=shear,
        bulk_modulus_pa=bulk,
        lame_lambda_pa=lame_lambda,
        volumetric_heat_capacity_j_m3k=volumetric_C,
    )


def verify_catalog_elastic_consistency(
    young_modulus_pa: float,
    derived_shear_pa: float,
    derived_bulk_pa: float,
    *,
    relative_tolerance: float = 0.01,
) -> bool:
    """Catalog sanity check: E should match 9 K mu / (3 K + mu) within 1%.

    Returns True if consistent, False otherwise. Callers (catalog loader)
    should log a warning on False, but not refuse to start.
    """
    denom = 3.0 * derived_bulk_pa + derived_shear_pa
    if denom <= 0:
        return False
    e_implied = 9.0 * derived_bulk_pa * derived_shear_pa / denom
    if young_modulus_pa == 0:
        return e_implied == 0
    return abs(e_implied - young_modulus_pa) / abs(young_modulus_pa) < relative_tolerance
