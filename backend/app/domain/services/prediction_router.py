from __future__ import annotations

from typing import Protocol

from app.core.exceptions import DomainValidationError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType


class PredictionClientProtocol(Protocol):
    model_type: ModelType

    async def predict(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse: ...

    def descriptor(self) -> dict[str, str]: ...


class PredictionRouter:
    def __init__(self, clients: list[PredictionClientProtocol]) -> None:
        self.clients = {client.model_type: client for client in clients}

    async def route(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse:
        client = self.clients.get(request.model)
        if client is None:
            raise DomainValidationError(
                code="UNKNOWN_MODEL",
                message=f"Unsupported model: {request.model.value}",
                details={"model": request.model.value},
            )
        return await client.predict(request)

    def list_models(self) -> list[dict[str, str]]:
        descriptors = {client.model_type: client.descriptor() for client in self.clients.values()}
        results: list[dict[str, str]] = []
        for model_type in ModelType:
            results.append(
                descriptors.get(
                    model_type,
                    {"id": model_type.value, "name": model_type.label, "status": "not_configured"},
                )
            )
        return results
