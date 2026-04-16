from __future__ import annotations

from typing import Any

from app.domain.entities.prediction import EnrichedPredictionRequest
from app.domain.enums.model_type import ModelType
from app.infrastructure.clients.base import BaseModelClient


class PINNClient(BaseModelClient):
    def __init__(self, base_url: str, predict_path: str, timeout_seconds: float) -> None:
        super().__init__(
            model_type=ModelType.PINN,
            service_name="PINN",
            base_url=base_url,
            predict_path=predict_path,
            timeout_seconds=timeout_seconds,
        )

    def build_payload(self, request: EnrichedPredictionRequest) -> dict[str, Any]:
        payload = request.to_shared_payload()
        payload["representation"] = "physics_informed"
        payload["routing_hint"] = "pinn"
        return payload
