"""Smoke test for the v2 media repository.

Loads the real catalog_v2.json and asserts the high-level invariants we
care about — 10 materials, 4 thermoelastic_supported (those that have
alpha_1_K in the CSV), 6 unsupported.
"""
from __future__ import annotations

from pathlib import Path

from app.infrastructure.repositories.media_repository_v2 import (
    MediaRepositoryV2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_V2 = PROJECT_ROOT / "backend" / "data" / "media" / "catalog_v2.json"


def test_catalog_v2_loads_all_10_materials():
    repo = MediaRepositoryV2(CATALOG_V2)
    media = repo.list_media()
    assert len(media) == 10


def test_catalog_v2_thermoelastic_split_matches_csv():
    repo = MediaRepositoryV2(CATALOG_V2)
    media = repo.list_media()
    supported = {m.id for m in media if m.thermoelastic_supported}
    unsupported = {m.id for m in media if not m.thermoelastic_supported}
    # The four materials that have alpha_1_K in the CSV
    assert supported == {"granite", "sandstone", "limestone", "marble"}
    # The six that don't
    assert unsupported == {
        "granodiorite", "basalt", "diabase", "gabbro", "schist", "quartzite",
    }


def test_catalog_v2_get_by_id_returns_filled_properties():
    repo = MediaRepositoryV2(CATALOG_V2)
    granite = repo.get_by_id("granite")
    assert granite is not None
    assert granite.thermoelastic_supported is True
    p = granite.properties
    assert p.rho_kg_m3 is not None and p.rho_kg_m3 > 0
    assert p.vp_m_s is not None and p.vp_m_s > 0
    assert p.young_modulus_pa is not None
    assert p.thermal_expansion_1_k is not None


def test_catalog_v2_unsupported_entry_has_alpha_none():
    repo = MediaRepositoryV2(CATALOG_V2)
    basalt = repo.get_by_id("basalt")
    assert basalt is not None
    assert basalt.thermoelastic_supported is False
    assert basalt.properties.thermal_expansion_1_k is None
    assert basalt.metadata.limitation is not None
