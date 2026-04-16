from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    api_prefix: str
    log_level: str
    media_catalog_path: Path
    cors_origins: list[str]
    remote_model_timeout_seconds: float
    model_meshgraphnet_url: str
    model_fno_url: str
    model_pinn_url: str
    model_meshgraphnet_predict_path: str
    model_fno_predict_path: str
    model_pinn_predict_path: str


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def get_settings() -> Settings:
    media_catalog_default = BASE_DIR / "data" / "media" / "catalog.json"
    return Settings(
        app_name=os.getenv("APP_NAME", "Thermoelastic Direction API"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        media_catalog_path=Path(os.getenv("MEDIA_CATALOG_PATH", str(media_catalog_default))).resolve(),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS"), ["*"]),
        remote_model_timeout_seconds=float(os.getenv("REMOTE_MODEL_TIMEOUT_SECONDS", "12")),
        model_meshgraphnet_url=os.getenv("MODEL_MESHGRAPHNET_URL", "http://localhost:9001"),
        model_fno_url=os.getenv("MODEL_FNO_URL", "http://localhost:9002"),
        model_pinn_url=os.getenv("MODEL_PINN_URL", "http://localhost:9003"),
        model_meshgraphnet_predict_path=os.getenv("MODEL_MESHGRAPHNET_PREDICT_PATH", "/predict"),
        model_fno_predict_path=os.getenv("MODEL_FNO_PREDICT_PATH", "/predict"),
        model_pinn_predict_path=os.getenv("MODEL_PINN_PREDICT_PATH", "/predict"),
    )
