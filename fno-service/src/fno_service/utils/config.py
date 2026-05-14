from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FNOServiceConfig:
    checkpoint_path: Path
    config_path: Path
    dataset_path: Path
    device: str
    log_level: str
    service_port: int
    allow_fallback: bool


def get_service_config() -> FNOServiceConfig:
    return FNOServiceConfig(
        checkpoint_path=Path(os.getenv("FNO_CHECKPOINT_PATH", "/app/artifacts/checkpoints/baseline")).expanduser(),
        config_path=Path(os.getenv("FNO_CONFIG_PATH", "/app/configs/inference.yaml")).expanduser(),
        dataset_path=Path(os.getenv("FNO_DATASET_PATH", "/app/artifacts/datasets/sandstone_fno")).expanduser(),
        device=os.getenv("FNO_DEVICE", "cpu"),
        log_level=os.getenv("FNO_LOG_LEVEL", "INFO"),
        service_port=int(os.getenv("FNO_SERVICE_PORT", "9000")),
        allow_fallback=_env_bool("FNO_ALLOW_FALLBACK", default=False),
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
