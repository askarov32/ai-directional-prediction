from __future__ import annotations

from typing import Protocol

from app.domain.entities.medium import Medium
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType


class MediumRepositoryPort(Protocol):
    def list_media(self) -> list[Medium]: ...

    def get_by_id(self, medium_id: str) -> Medium | None: ...


class ModelClientPort(Protocol):
    model_type: ModelType

    async def predict(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse: ...

    async def readiness(self) -> dict: ...

    def descriptor(self) -> dict[str, str]: ...


class PredictionResponseNormalizerPort(Protocol):
    def normalize(self, request: EnrichedPredictionRequest, remote: RemotePredictionResponse) -> dict: ...
