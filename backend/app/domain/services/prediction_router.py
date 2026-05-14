from __future__ import annotations

from app.core.exceptions import DomainValidationError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType
from app.domain.ports import ModelClientPort


MODEL_CAPABILITIES: dict[ModelType, dict[str, object]] = {
    ModelType.MESHGRAPHNET: {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "capability_note": "Graph-based route; use this for 3D-first experiments.",
    },
    ModelType.PINN: {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "capability_note": "Physics-informed route; recommended for 3D thermoelastic studies.",
    },
    ModelType.TRANSFORMER: {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "capability_note": "Sequence rollout route; 3D-ready in the current MVP stack.",
    },
    ModelType.FNO: {
        "supported_domain_types": ["rect_2d"],
        "default_domain_type": "rect_2d",
        "capability_note": "Current baseline is FNO2d, so 3D requests should use another model.",
    },
}


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
        results: list[dict[str, object]] = []
        for model_type in ModelType:
            capability = MODEL_CAPABILITIES.get(model_type, {})
            results.append(
                {
                    **descriptors.get(
                        model_type,
                        {"id": model_type.value, "name": model_type.label, "status": "not_configured"},
                    ),
                    "supported_domain_types": capability.get("supported_domain_types", []),
                    "default_domain_type": capability.get("default_domain_type"),
                    "capability_note": capability.get("capability_note"),
                }
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
