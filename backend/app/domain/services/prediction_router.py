from __future__ import annotations

from app.core.exceptions import DomainValidationError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType
from app.domain.ports import ModelClientPort


class PredictionRouter:
    def __init__(self, clients: list[ModelClientPort]) -> None:
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

    async def readiness(self) -> list[dict]:
        checks: list[dict] = []
        for model_type in ModelType:
            client = self.clients.get(model_type)
            if client is None:
                checks.append(
                    {
                        "id": model_type.value,
                        "name": model_type.label,
                        "ready": False,
                        "status": "not_configured",
                    }
                )
                continue
            checks.append(await client.readiness())
        return checks
