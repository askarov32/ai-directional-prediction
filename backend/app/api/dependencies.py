from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.domain.services.medium_catalog import MediumCatalogService
from app.domain.services.prediction_router import PredictionRouter
from app.domain.use_cases.predict_direction import PredictDirectionUseCase
from app.infrastructure.adapters.response_normalizer import ResponseNormalizer
from app.infrastructure.clients.fno_client import FNOClient
from app.infrastructure.clients.meshgraphnet_client import MeshGraphNetClient
from app.infrastructure.clients.pinn_client import PINNClient
from app.infrastructure.repositories.media_repository import MediaRepository


@lru_cache
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache
def get_media_repository() -> MediaRepository:
    settings = get_app_settings()
    return MediaRepository(settings.media_catalog_path)


@lru_cache
def get_medium_catalog_service() -> MediumCatalogService:
    return MediumCatalogService(get_media_repository())


@lru_cache
def get_response_normalizer() -> ResponseNormalizer:
    return ResponseNormalizer()


@lru_cache
def get_prediction_router() -> PredictionRouter:
    settings = get_app_settings()
    clients = [
        MeshGraphNetClient(
            base_url=settings.model_meshgraphnet_url,
            predict_path=settings.model_meshgraphnet_predict_path,
            timeout_seconds=settings.remote_model_timeout_seconds,
        ),
        FNOClient(
            base_url=settings.model_fno_url,
            predict_path=settings.model_fno_predict_path,
            timeout_seconds=settings.remote_model_timeout_seconds,
        ),
        PINNClient(
            base_url=settings.model_pinn_url,
            predict_path=settings.model_pinn_predict_path,
            timeout_seconds=settings.remote_model_timeout_seconds,
        ),
    ]
    return PredictionRouter(clients=clients)


@lru_cache
def get_predict_direction_use_case() -> PredictDirectionUseCase:
    return PredictDirectionUseCase(
        medium_catalog=get_medium_catalog_service(),
        prediction_router=get_prediction_router(),
        response_normalizer=get_response_normalizer(),
    )
