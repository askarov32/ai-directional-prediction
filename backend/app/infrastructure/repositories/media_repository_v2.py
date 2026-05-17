"""v2 media repository — reads backend/data/media/catalog_v2.json.

Parallel to ``MediaRepository`` (v1) so the v1 path keeps reading
``catalog.json``. Materials whose source table does not include
``alpha_1_K`` are surfaced with ``thermoelastic_supported=False``;
callers (the v2 use case) must reject them for thermoelastic requests.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.domain.entities.medium import (
    MediumMetadataV2,
    MediumPropertiesV2,
    MediumV2,
)


class MediaRepositoryV2:
    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        self._cache: list[MediumV2] | None = None

    def list_media(self) -> list[MediumV2]:
        if self._cache is None:
            self._cache = self._load()
        return list(self._cache)

    def get_by_id(self, medium_id: str) -> MediumV2 | None:
        return next(
            (m for m in self.list_media() if m.id == medium_id), None
        )

    def _load(self) -> list[MediumV2]:
        raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        media: list[MediumV2] = []
        for item in raw:
            props = item["properties"]
            meta = item["metadata"]
            media.append(
                MediumV2(
                    id=item["id"],
                    name=item["name"],
                    category=item.get("category", ""),
                    thermoelastic_supported=bool(item["thermoelastic_supported"]),
                    properties=MediumPropertiesV2(
                        rho_kg_m3=props.get("rho_kg_m3"),
                        vp_m_s=props.get("vp_m_s"),
                        vs_m_s=props.get("vs_m_s"),
                        young_modulus_pa=props.get("young_modulus_pa"),
                        poisson_ratio=props.get("poisson_ratio"),
                        shear_modulus_pa=props.get("shear_modulus_pa"),
                        bulk_modulus_pa=props.get("bulk_modulus_pa"),
                        lame_lambda_pa=props.get("lame_lambda_pa"),
                        thermal_conductivity_w_mk=props.get(
                            "thermal_conductivity_w_mk"
                        ),
                        heat_capacity_j_kgk=props.get("heat_capacity_j_kgk"),
                        volumetric_heat_capacity_j_m3k=props.get(
                            "volumetric_heat_capacity_j_m3k"
                        ),
                        thermal_expansion_1_k=props.get("thermal_expansion_1_k"),
                        thermoelastic_gamma_pa_k=props.get(
                            "thermoelastic_gamma_pa_k"
                        ),
                        porosity_summary=props.get("porosity_summary"),
                    ),
                    metadata=MediumMetadataV2(
                        source_table=meta.get("source_table", ""),
                        value_type=meta.get("value_type", ""),
                        source_files=meta.get("source_files", ""),
                        notes=meta.get("notes", ""),
                        limitation=meta.get("limitation"),
                    ),
                )
            )
        return media
