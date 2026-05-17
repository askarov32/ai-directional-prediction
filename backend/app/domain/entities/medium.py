from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MediumProperties:
    rho: float
    porosity_total: float
    porosity_effective: float
    vp: float
    vs: float
    thermal_conductivity: float
    heat_capacity: float
    thermal_expansion: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rho": self.rho,
            "porosity_total": self.porosity_total,
            "porosity_effective": self.porosity_effective,
            "vp": self.vp,
            "vs": self.vs,
            "thermal_conductivity": self.thermal_conductivity,
            "heat_capacity": self.heat_capacity,
            "thermal_expansion": self.thermal_expansion,
        }


@dataclass(frozen=True)
class MediumRanges:
    temperature_c: tuple[float, float]
    pressure_mpa: tuple[float, float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "temperature_c": [self.temperature_c[0], self.temperature_c[1]],
            "pressure_mpa": [self.pressure_mpa[0], self.pressure_mpa[1]],
        }


@dataclass(frozen=True)
class MediumMetadata:
    source: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "notes": self.notes}


@dataclass(frozen=True)
class Medium:
    id: str
    name: str
    category: str
    properties: MediumProperties
    ranges: MediumRanges
    metadata: MediumMetadata

    def summary(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "properties": self.properties.to_dict(),
            "ranges": self.ranges.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


# ---------------------------------------------------------------------------
# v2 medium (additive). catalog_v2.json is a projection of
# chapters/tables/combined_geological_media_parameters.csv.
# Materials without alpha_1_K carry thermoelastic_supported=False.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediumPropertiesV2:
    rho_kg_m3: float | None
    vp_m_s: float | None
    vs_m_s: float | None
    young_modulus_pa: float | None
    poisson_ratio: float | None
    shear_modulus_pa: float | None
    bulk_modulus_pa: float | None
    lame_lambda_pa: float | None
    thermal_conductivity_w_mk: float | None
    heat_capacity_j_kgk: float | None
    volumetric_heat_capacity_j_m3k: float | None
    thermal_expansion_1_k: float | None
    thermoelastic_gamma_pa_k: float | None
    porosity_summary: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho_kg_m3": self.rho_kg_m3,
            "vp_m_s": self.vp_m_s,
            "vs_m_s": self.vs_m_s,
            "young_modulus_pa": self.young_modulus_pa,
            "poisson_ratio": self.poisson_ratio,
            "shear_modulus_pa": self.shear_modulus_pa,
            "bulk_modulus_pa": self.bulk_modulus_pa,
            "lame_lambda_pa": self.lame_lambda_pa,
            "thermal_conductivity_w_mk": self.thermal_conductivity_w_mk,
            "heat_capacity_j_kgk": self.heat_capacity_j_kgk,
            "volumetric_heat_capacity_j_m3k": self.volumetric_heat_capacity_j_m3k,
            "thermal_expansion_1_k": self.thermal_expansion_1_k,
            "thermoelastic_gamma_pa_k": self.thermoelastic_gamma_pa_k,
            "porosity_summary": self.porosity_summary,
        }


@dataclass(frozen=True)
class MediumMetadataV2:
    source_table: str
    value_type: str
    source_files: str
    notes: str
    limitation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_table": self.source_table,
            "value_type": self.value_type,
            "source_files": self.source_files,
            "notes": self.notes,
        }
        if self.limitation is not None:
            out["limitation"] = self.limitation
        return out


@dataclass(frozen=True)
class MediumV2:
    id: str
    name: str
    category: str
    thermoelastic_supported: bool
    properties: MediumPropertiesV2
    metadata: MediumMetadataV2

    def summary(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "category": self.category}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "thermoelastic_supported": self.thermoelastic_supported,
            "properties": self.properties.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
