from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import httpx

from app.core.exceptions import MalformedRemoteResponseError, RemoteServiceTimeoutError, RemoteServiceUnavailableError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.domain.enums.model_type import ModelType


class BaseModelClient(ABC):
    def __init__(
        self,
        model_type: ModelType,
        service_name: str,
        base_url: str,
        predict_path: str,
        timeout_seconds: float,
    ) -> None:
        self.model_type = model_type
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.predict_path = predict_path if predict_path.startswith("/") else f"/{predict_path}"
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"app.infrastructure.clients.{self.service_name.lower()}")

    @abstractmethod
    def build_payload(self, request: EnrichedPredictionRequest) -> dict[str, Any]:
        raise NotImplementedError

    async def predict(self, request: EnrichedPredictionRequest) -> RemotePredictionResponse:
        payload = self.build_payload(request)
        url = f"{self.base_url}{self.predict_path}"
        started_at = perf_counter()
        self.logger.info("Calling remote model service %s", self.service_name)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise RemoteServiceTimeoutError(self.service_name, {"url": url}) from exc
        except httpx.ConnectError as exc:
            raise RemoteServiceUnavailableError(self.service_name, {"url": url}) from exc
        except httpx.HTTPStatusError as exc:
            raise RemoteServiceUnavailableError(
                self.service_name,
                {"url": url, "status_code": exc.response.status_code},
            ) from exc
        except ValueError as exc:
            raise MalformedRemoteResponseError(self.service_name, {"url": url}) from exc

        if not isinstance(data, dict):
            raise MalformedRemoteResponseError(self.service_name, {"url": url, "reason": "response is not a JSON object"})

        latency_ms = int((perf_counter() - started_at) * 1000)
        return RemotePredictionResponse(service_name=self.service_name, payload=data, latency_ms=latency_ms)

    def descriptor(self) -> dict[str, str]:
        return {
            "id": self.model_type.value,
            "name": self.model_type.label,
            "status": "configured" if self.base_url else "not_configured",
        }
