from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path
    device: str
    log_level: str
    service_port: int
    reference_temperature_k: float
    time_scale: float


def get_inference_config() -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path=Path(
            os.getenv("PINN_CHECKPOINT_PATH", "/app/artifacts/checkpoints/baseline/model.pth")
        ).expanduser(),
        device=os.getenv("PINN_DEVICE", "cpu"),
        log_level=os.getenv("PINN_LOG_LEVEL", "INFO"),
        service_port=int(os.getenv("PINN_SERVICE_PORT", "9000")),
        reference_temperature_k=float(os.getenv("PINN_REFERENCE_TEMPERATURE_K", "293.15")),
        time_scale=float(os.getenv("PINN_TIME_SCALE", "1.0")),
    )
