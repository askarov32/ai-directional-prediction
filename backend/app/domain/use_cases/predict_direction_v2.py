"""v2 prediction use case.

Orchestration only — no IO, no FastAPI. Given a parsed v2 request,
resolves the medium from catalog_v2, enforces thermoelastic support,
builds a v1-shaped enriched payload for the existing router (so the
model services keep working untouched in Phase 2), waits for the remote
response, and normalises it into the v2 contract shape.

Phase 4 will swap the v1 router for one that speaks v2 natively; until
then this is the compatibility seam.
"""
from __future__ import annotations

from time import perf_counter

from app.domain.entities.medium import (
    Medium,
    MediumMetadata,
    MediumProperties,
    MediumRanges,
    MediumV2,
)
from app.domain.entities.prediction import (
    BoundaryConditions,
    Domain,
    DomainResolution,
    DomainSize,
    EnrichedPredictionRequest,
    Probe,
    Scenario,
    Source,
    UnifiedPredictionRequestV2,
)
from app.domain.services.derived_quantities import (
    DOMAIN_SIZE_M,
    REFERENCE_TEMPERATURE_K,
    SOURCE_TEMPERATURE_K,
    compute_derived_geometry,
)
from app.domain.services.medium_catalog_v2 import MediumCatalogServiceV2
from app.domain.services.prediction_router import PredictionRouter
from app.infrastructure.adapters.response_normalizer_v2 import (
    DerivedGeometry2D,
    normalize_to_v2,
)


# Legacy enrichment defaults — used only to satisfy the v1 EnrichedPredictionRequest
# contract while Phase 4 hasn't migrated the model clients yet.
_LEGACY_FREQUENCY_HZ = 25.0
_LEGACY_AMPLITUDE = 1.0
_LEGACY_SOURCE_TYPE = "thermal_pulse"
_DEFAULT_PRESSURE_MPA = 5.0


def _convert_to_v1_medium(medium: MediumV2) -> Medium:
    """Project a v2 medium back to the v1 dataclass for the legacy router."""
    props = medium.properties
    # v1 keys: rho, vp, vs, thermal_conductivity, heat_capacity,
    # thermal_expansion, porosity_total, porosity_effective.
    # CSV stores vp/vs in m/s; v1 dataclass historically held km/s for
    # vp/vs but they round-trip into model service payloads anyway.
    vp_km_s = (props.vp_m_s or 0.0) / 1000.0
    vs_km_s = (props.vs_m_s or 0.0) / 1000.0
    return Medium(
        id=medium.id,
        name=medium.name,
        category=medium.category,
        properties=MediumProperties(
            rho=float(props.rho_kg_m3 or 0.0),
            porosity_total=0.0,  # not modelled in v2; safe default
            porosity_effective=0.0,
            vp=vp_km_s,
            vs=vs_km_s,
            thermal_conductivity=float(props.thermal_conductivity_w_mk or 0.0),
            heat_capacity=float(props.heat_capacity_j_kgk or 0.0),
            thermal_expansion=float(props.thermal_expansion_1_k or 0.0),
        ),
        ranges=MediumRanges(
            temperature_c=(-273.15, 2000.0),
            pressure_mpa=(0.1, 5000.0),
        ),
        metadata=MediumMetadata(
            source=medium.metadata.source_table,
            notes=medium.metadata.notes,
        ),
    )


def _build_legacy_enriched(
    request: UnifiedPredictionRequestV2, v1_medium: Medium,
    derived_geometry: DerivedGeometry2D,
) -> EnrichedPredictionRequest:
    """Pack a v2 request into the v1 EnrichedPredictionRequest the
    existing router/clients expect. The v1 model services do not see
    `thermal_state` directly; we encode source temperature as
    `scenario.temperature_c` (= θ + 0 °C) and leave everything else at
    safe defaults derived from the locked v2 invariants."""
    theta_k = (
        request.thermal_state.source_temperature_k
        - request.thermal_state.reference_temperature_k
    )
    return EnrichedPredictionRequest(
        model=request.model,
        medium=v1_medium,
        scenario=Scenario(
            temperature_c=theta_k,  # delta from 0 C (v1 used Celsius)
            pressure_mpa=_DEFAULT_PRESSURE_MPA,
            time_ms=request.observation.time_s * 1000.0,
        ),
        source=Source(
            type=_LEGACY_SOURCE_TYPE,
            x=request.geometry.source.x_m,
            y=request.geometry.source.y_m,
            z=0.0,
            amplitude=_LEGACY_AMPLITUDE,
            frequency_hz=_LEGACY_FREQUENCY_HZ,
            direction=(
                derived_geometry.unit_direction[0],
                derived_geometry.unit_direction[1],
                0.0,
            ),
        ),
        probe=Probe(
            x=request.geometry.probe.x_m,
            y=request.geometry.probe.y_m,
            z=0.0,
        ),
        domain=Domain(
            type="rect_2d",
            size=DomainSize(lx=DOMAIN_SIZE_M, ly=DOMAIN_SIZE_M, lz=0.0),
            resolution=DomainResolution(nx=128, ny=128, nz=1),
            boundary_conditions=BoundaryConditions(
                left="fixed", right="free", top="insulated", bottom="insulated"
            ),
        ),
    )


class PredictDirectionV2UseCase:
    def __init__(
        self,
        medium_catalog: MediumCatalogServiceV2,
        prediction_router: PredictionRouter,
    ) -> None:
        self.medium_catalog = medium_catalog
        self.prediction_router = prediction_router

    async def execute(self, request: UnifiedPredictionRequestV2) -> dict:
        medium_v2 = self.medium_catalog.get_medium(request.medium_id)
        self.medium_catalog.require_thermoelastic_support(medium_v2)

        derived_geo = compute_derived_geometry(
            (request.geometry.source.x_m, request.geometry.source.y_m),
            (request.geometry.probe.x_m, request.geometry.probe.y_m),
        )
        derived_geo_dc = DerivedGeometry2D(
            propagation_vector_m=derived_geo.propagation_vector_m,
            distance_m=derived_geo.distance_m,
            unit_direction=derived_geo.unit_direction,
            azimuth_deg=derived_geo.azimuth_deg,
        )

        v1_medium = _convert_to_v1_medium(medium_v2)
        legacy_request = _build_legacy_enriched(request, v1_medium, derived_geo_dc)

        started = perf_counter()
        remote = await self.prediction_router.route(legacy_request)
        elapsed_ms = (perf_counter() - started) * 1000.0

        return normalize_to_v2(
            request,
            medium_v2,
            derived_geo_dc,
            remote.payload,
            route="/predict",
            inference_time_ms=elapsed_ms,
        )
