from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_medium_catalog_service
from app.domain.services.medium_catalog import MediumCatalogService
from app.schemas.media import MediumResponseSchema

router = APIRouter(tags=["media"])


@router.get("/media", response_model=list[MediumResponseSchema])
async def list_media(
    medium_catalog: MediumCatalogService = Depends(get_medium_catalog_service),
) -> list[MediumResponseSchema]:
    return [MediumResponseSchema.from_entity(medium) for medium in medium_catalog.list_media()]


@router.get("/media/{medium_id}", response_model=MediumResponseSchema)
async def get_medium(
    medium_id: str,
    medium_catalog: MediumCatalogService = Depends(get_medium_catalog_service),
) -> MediumResponseSchema:
    medium = medium_catalog.get_medium(medium_id)
    return MediumResponseSchema.from_entity(medium)
