from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fno_service.api.schemas import PredictionPayload
from fno_service.utils.config import FNOServiceConfig


class CheckpointNotReadyError(RuntimeError):
    """Raised when FNO inference cannot run because no checkpoint is ready."""


class FNOInferenceService:
    def __init__(self, config: FNOServiceConfig) -> None:
        self.config = config

    def health_payload(self) -> dict[str, Any]:
        checkpoint = self._resolve_checkpoint_path()
        return {
            "status": "ok",
            "service": "fno-service",
            "ready": self._is_ready(),
            "mode": self._mode(),
            "checkpoint_path": str(self.config.checkpoint_path),
            "resolved_checkpoint_path": str(checkpoint) if checkpoint else None,
            "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
            "dataset_path": str(self.config.dataset_path),
            "config_path": str(self.config.config_path),
            "device": self.config.device,
            "allow_fallback": self.config.allow_fallback,
        }

    def readiness_payload(self) -> dict[str, Any]:
        payload = self.health_payload()
        payload["status"] = "ready" if payload["ready"] else "not_ready"
        return payload

    def predict(self, payload: PredictionPayload) -> dict[str, Any]:
        if not self._is_ready():
            raise CheckpointNotReadyError(
                f"FNO checkpoint is not ready at {self.config.checkpoint_path}. "
                "Train a checkpoint or set FNO_ALLOW_FALLBACK=true for local skeleton wiring."
            )
        return self._fallback_prediction(payload)

    def _is_ready(self) -> bool:
        checkpoint = self._resolve_checkpoint_path()
        return bool(checkpoint and checkpoint.exists()) or self.config.allow_fallback

    def _mode(self) -> str:
        checkpoint = self._resolve_checkpoint_path()
        if checkpoint and checkpoint.exists():
            return "checkpoint"
        if self.config.allow_fallback:
            return "fallback"
        return "not_ready"

    def _resolve_checkpoint_path(self) -> Path | None:
        path = self.config.checkpoint_path
        if path.is_file():
            return path
        if path.is_dir():
            for name in ("best_model.pth", "model.pth"):
                candidate = path / name
                if candidate.exists():
                    return candidate
        return path

    def _fallback_prediction(self, payload: PredictionPayload) -> dict[str, Any]:
        source = payload.source
        probe = payload.probe
        scenario = payload.scenario
        medium = payload.medium
        properties = medium.get("properties", {}) if isinstance(medium, dict) else {}

        dx = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
        dy = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
        dz = float(probe.get("z", 0.0)) - float(source.get("z", 0.0))
        direction = _normalize([dx + 0.06, dy - 0.03, dz])

        azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        horizontal = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        elevation = math.degrees(math.atan2(direction[2], horizontal))
        distance = math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-8))
        vp = float(properties.get("vp", 5.0) or 5.0)
        temperature = float(scenario.get("temperature_c", 20.0))
        pressure = float(scenario.get("pressure_mpa", 1.0))
        time_ms = float(scenario.get("time_ms", 1.0))

        return {
            "prediction": {
                "direction_vector": [round(value, 6) for value in direction],
                "azimuth_deg": round(azimuth, 4),
                "elevation_deg": round(elevation, 4),
                "magnitude": 1.0,
                "wave_type": "fno_skeleton_fallback",
                "travel_time_ms": round(max((distance / max(vp, 0.1)) * 10.0 + time_ms * 0.12, 0.001), 6),
            },
            "field_summary": {
                "max_displacement": round(0.001 + temperature / 350000.0 + pressure / 300000.0, 8),
                "max_temperature_perturbation": round(max(temperature, 0.0) / 160.0 + 0.45, 8),
            },
            "model_version": "fno-skeleton-fallback-v0",
            "diagnostics": {
                "mode": self._mode(),
                "service_config": asdict(self.config),
                "note": "Phase 2 skeleton response. Real FNO checkpoint inference is not implemented yet.",
            },
        }


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return [1.0, 0.0, 0.0]
    return [component / magnitude for component in vector]
