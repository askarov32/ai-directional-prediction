from __future__ import annotations

import math
from uuid import uuid4

from app.core.exceptions import MalformedRemoteResponseError
from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse


class ResponseNormalizer:
    def normalize(self, request: EnrichedPredictionRequest, remote: RemotePredictionResponse) -> dict:
        raw = remote.payload
        prediction = raw.get("prediction", raw)
        field_summary = raw.get("field_summary", raw)
        direction_vector = self._coerce_vector(prediction.get("direction_vector"), remote.service_name)
        magnitude = prediction.get("magnitude", self._vector_magnitude(direction_vector))

        return {
            "model": request.model.value,
            "medium": request.medium.summary(),
            "prediction": {
                "direction_vector": direction_vector,
                "azimuth_deg": round(float(prediction.get("azimuth_deg", self._calculate_azimuth(direction_vector))), 2),
                "elevation_deg": round(
                    float(prediction.get("elevation_deg", self._calculate_elevation(direction_vector))),
                    2,
                ),
                "magnitude": round(float(magnitude), 4),
                "wave_type": str(prediction.get("wave_type", "unknown")),
                "travel_time_ms": round(float(prediction.get("travel_time_ms", 0.0)), 3),
            },
            "field_summary": {
                "max_displacement": round(float(field_summary.get("max_displacement", 0.0)), 6),
                "max_temperature_perturbation": round(
                    float(field_summary.get("max_temperature_perturbation", 0.0)),
                    6,
                ),
            },
            "meta": {
                "model_version": str(raw.get("model_version") or raw.get("meta", {}).get("model_version", "unknown")),
                "latency_ms": remote.latency_ms,
                "request_id": str(uuid4()),
            },
        }

    def _coerce_vector(self, value: object, service_name: str) -> list[float]:
        if not isinstance(value, list) or len(value) != 3:
            raise MalformedRemoteResponseError(
                service_name,
                {"reason": "direction_vector must be a list of length 3"},
            )
        try:
            vector = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise MalformedRemoteResponseError(service_name, {"reason": "direction_vector contains non-numeric values"}) from exc
        return vector

    def _vector_magnitude(self, vector: list[float]) -> float:
        return math.sqrt(sum(component * component for component in vector))

    def _calculate_azimuth(self, vector: list[float]) -> float:
        return math.degrees(math.atan2(vector[1], vector[0]))

    def _calculate_elevation(self, vector: list[float]) -> float:
        horizontal = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        return math.degrees(math.atan2(vector[2], horizontal))
