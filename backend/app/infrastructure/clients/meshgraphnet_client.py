from __future__ import annotations

from typing import Any

from app.domain.entities.prediction import EnrichedPredictionRequest
from app.domain.enums.model_type import ModelType
from app.infrastructure.clients.base import BaseModelClient


class MeshGraphNetClient(BaseModelClient):
    def __init__(self, base_url: str, predict_path: str, timeout_seconds: float) -> None:
        super().__init__(
            model_type=ModelType.MESHGRAPHNET,
            service_name="MeshGraphNet",
            base_url=base_url,
            predict_path=predict_path,
            timeout_seconds=timeout_seconds,
        )

    def build_payload(self, request: EnrichedPredictionRequest) -> dict[str, Any]:
        payload = request.to_shared_payload()
        payload["representation"] = "graph"
        payload["routing_hint"] = "meshgraphnet"
        return payload
