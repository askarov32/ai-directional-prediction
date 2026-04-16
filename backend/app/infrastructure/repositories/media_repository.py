from __future__ import annotations

import json
from pathlib import Path

from app.domain.entities.medium import Medium, MediumMetadata, MediumProperties, MediumRanges


class MediaRepository:
    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        self._cache: list[Medium] | None = None

    def list_media(self) -> list[Medium]:
        if self._cache is None:
            self._cache = self._load()
        return list(self._cache)

    def get_by_id(self, medium_id: str) -> Medium | None:
        return next((medium for medium in self.list_media() if medium.id == medium_id), None)

    def _load(self) -> list[Medium]:
        raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        media: list[Medium] = []
        for item in raw:
            media.append(
                Medium(
                    id=item["id"],
                    name=item["name"],
                    category=item["category"],
                    properties=MediumProperties(**item["properties"]),
                    ranges=MediumRanges(
                        temperature_c=tuple(item["ranges"]["temperature_c"]),
                        pressure_mpa=tuple(item["ranges"]["pressure_mpa"]),
                    ),
                    metadata=MediumMetadata(**item["metadata"]),
                )
            )
        return media
