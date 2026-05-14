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
    rollout_steps: int


def get_inference_config() -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path=Path(
            os.getenv(
                "TRANSFORMER_CHECKPOINT_PATH",
                "/app/artifacts/checkpoints/baseline",
            )
        ).expanduser(),
        device=os.getenv("TRANSFORMER_DEVICE", "cpu"),
        log_level=os.getenv("TRANSFORMER_LOG_LEVEL", "INFO"),
        service_port=int(os.getenv("TRANSFORMER_SERVICE_PORT", "9000")),
        reference_temperature_k=float(
            os.getenv("TRANSFORMER_REFERENCE_TEMPERATURE_K", "293.15")
        ),
        time_scale=float(os.getenv("TRANSFORMER_TIME_SCALE", "1.0")),
        rollout_steps=int(os.getenv("TRANSFORMER_ROLLOUT_STEPS", "100")),
    )
