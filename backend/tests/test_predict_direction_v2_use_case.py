"""End-to-end test of the v2 use case with fake catalog + fake router.

Verifies that:
- thermoelastically-unsupported materials are rejected
- unknown materials raise ResourceNotFoundError
- happy path returns a v2-shaped dict that validates against
  ``PredictionResponseV2Schema``
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.core.exceptions import DomainValidationError, ResourceNotFoundError
from app.domain.entities.medium import (
    MediumMetadataV2,
    MediumPropertiesV2,
    MediumV2,
)
from app.domain.entities.prediction import (
    EnrichedPredictionRequest,
    Geometry2D,
    ObservationV2,
    Point2D,
    RemotePredictionResponse,
    ScenarioPrototypeV2,
    ThermalStateV2,
    UnifiedPredictionRequestV2,
)
from app.domain.enums.model_type import ModelType
from app.domain.services.derived_quantities import (
    REFERENCE_TEMPERATURE_K,
    SOURCE_TEMPERATURE_K,
)
from app.domain.services.medium_catalog_v2 import MediumCatalogServiceV2
from app.domain.use_cases.predict_direction_v2 import PredictDirectionV2UseCase
from app.schemas.prediction import PredictionResponseV2Schema


class _FakeRepo:
    def __init__(self, media: list[MediumV2]) -> None:
        self._media = media

    def list_media(self) -> list[MediumV2]:
        return list(self._media)

    def get_by_id(self, medium_id: str) -> MediumV2 | None:
        return next((m for m in self._media if m.id == medium_id), None)


class _FakeRouter:
    """Mimics PredictionRouter.route by returning a canned remote response.

    Captures the EnrichedPredictionRequest the use case built so the test
    can assert on derived geometry / unit conversions if needed.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_request: EnrichedPredictionRequest | None = None

    async def route(
        self, request: EnrichedPredictionRequest
    ) -> RemotePredictionResponse:
        self.last_request = request
        return RemotePredictionResponse(
            service_name=request.model.value,
            payload=self.payload,
            latency_ms=5,
        )


def _medium_granite(thermoelastic: bool = True) -> MediumV2:
    return MediumV2(
        id="granite",
        name="Granite",
        category="igneous intrusive",
        thermoelastic_supported=thermoelastic,
        properties=MediumPropertiesV2(
            rho_kg_m3=2650.0, vp_m_s=5850.0, vs_m_s=3400.0,
            young_modulus_pa=7.6e10, poisson_ratio=0.245,
            shear_modulus_pa=3.0e10, bulk_modulus_pa=5.0e10,
            lame_lambda_pa=2.9e10,
            thermal_conductivity_w_mk=2.5,
            heat_capacity_j_kgk=850.0,
            volumetric_heat_capacity_j_m3k=2.25e6,
            thermal_expansion_1_k=7.9e-6 if thermoelastic else None,
            thermoelastic_gamma_pa_k=None,
            porosity_summary=None,
        ),
        metadata=MediumMetadataV2(
            source_table="combined_geological_media_parameters.csv",
            value_type="mixed", source_files="", notes="",
            limitation=None if thermoelastic else "no alpha",
        ),
    )


def _make_request() -> UnifiedPredictionRequestV2:
    return UnifiedPredictionRequestV2(
        model=ModelType.PINN,
        medium_id="granite",
        geometry=Geometry2D(
            dimension=2,
            source=Point2D(x_m=0.2, y_m=0.5),
            probe=Point2D(x_m=0.8, y_m=0.5),
        ),
        observation=ObservationV2(time_s=0.1),
        scenario=ScenarioPrototypeV2(
            thermal_source_type="point",
            mechanical_constraint="free",
            boundary_condition_type="prototype_simplified",
        ),
        thermal_state=ThermalStateV2(
            reference_temperature_k=REFERENCE_TEMPERATURE_K,
            source_temperature_k=SOURCE_TEMPERATURE_K,
        ),
    )


def _canned_remote() -> dict[str, Any]:
    return {
        "direction_vector": [1.0, 0.0, 0.0],
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "magnitude": 0.035,
        "wave_type": "physics_informed",
        "travel_time_ms": 0.1,
        "max_displacement": 1.2e-5,
        "max_temperature_perturbation": 0.05,
        "model_version": "pinn-baseline@best",
        "model_outputs": {
            "feature_names": ["temperature_k", "disp_x", "disp_y", "disp_z"],
            "values": [293.15, -1.1e-8, -1.2e-8, 2.2e-5],
        },
    }


@pytest.mark.asyncio
async def test_happy_path_returns_v2_validated_response():
    catalog = MediumCatalogServiceV2(_FakeRepo([_medium_granite()]))
    router = _FakeRouter(_canned_remote())
    use_case = PredictDirectionV2UseCase(catalog, router)

    out = await use_case.execute(_make_request())

    # Must round-trip cleanly through the v2 Pydantic response schema
    PredictionResponseV2Schema.model_validate(out)

    # Use case must have called the router with the converted v1 request
    assert router.last_request is not None
    assert router.last_request.model == ModelType.PINN
    assert router.last_request.medium.id == "granite"
    # v2 fixed source temperature = 1500 K. theta = 1226.85 K
    assert router.last_request.scenario.temperature_c == pytest.approx(1226.85)
    # observation 0.1 s -> 100 ms
    assert router.last_request.scenario.time_ms == pytest.approx(100.0)
    # geometry passed through
    assert router.last_request.source.x == pytest.approx(0.2)
    assert router.last_request.probe.x == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_rejects_thermoelastically_unsupported_material():
    catalog = MediumCatalogServiceV2(
        _FakeRepo([_medium_granite(thermoelastic=False)])
    )
    router = _FakeRouter(_canned_remote())
    use_case = PredictDirectionV2UseCase(catalog, router)

    with pytest.raises(DomainValidationError) as excinfo:
        await use_case.execute(_make_request())
    assert excinfo.value.code == "material_thermoelastic_unsupported"
    # router must never have been called
    assert router.last_request is None


@pytest.mark.asyncio
async def test_rejects_unknown_medium():
    catalog = MediumCatalogServiceV2(_FakeRepo([]))
    router = _FakeRouter(_canned_remote())
    use_case = PredictDirectionV2UseCase(catalog, router)
    with pytest.raises(ResourceNotFoundError):
        await use_case.execute(_make_request())


@pytest.mark.asyncio
async def test_request_id_present_and_unique():
    catalog = MediumCatalogServiceV2(_FakeRepo([_medium_granite()]))
    router = _FakeRouter(_canned_remote())
    use_case = PredictDirectionV2UseCase(catalog, router)
    a = await use_case.execute(_make_request())
    b = await use_case.execute(_make_request())
    assert a["request_id"] and b["request_id"]
    assert a["request_id"] != b["request_id"]
