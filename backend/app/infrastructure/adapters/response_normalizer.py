from __future__ import annotations

from uuid import uuid4

from app.domain.entities.prediction import EnrichedPredictionRequest, RemotePredictionResponse
from app.infrastructure.adapters.remote_response_schema import RemoteModelResponse


class ResponseNormalizer:
    def normalize(self, request: EnrichedPredictionRequest, remote: RemotePredictionResponse) -> dict:
        remote_response = RemoteModelResponse.from_payload(remote.payload, remote.service_name)
        prediction = remote_response.prediction
        field_summary = remote_response.field_summary

        return {
            "model": request.model.value,
            "medium": request.medium.summary(),
            "prediction": {
                "direction_vector": prediction.direction_vector,
                "azimuth_deg": round(prediction.azimuth_deg, 2),
                "elevation_deg": round(prediction.elevation_deg, 2),
                "magnitude": round(prediction.magnitude, 4),
                "wave_type": prediction.wave_type,
                "travel_time_ms": round(prediction.travel_time_ms, 3),
            },
            "field_summary": {
                "max_displacement": round(field_summary.max_displacement, 6),
                "max_temperature_perturbation": round(field_summary.max_temperature_perturbation, 6),
            },
            "meta": {
                "model_version": remote_response.model_version,
                "latency_ms": remote.latency_ms,
                "request_id": str(uuid4()),
            },
        }
