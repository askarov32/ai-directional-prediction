from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        protected_namespaces=("settings_",),
        validate_default=True,
    )

    app_name: str = "Thermoelastic Direction API"
    app_version: str = "0.1.0"
    environment: Literal["local", "development", "test", "production"] = "local"
    api_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    media_catalog_path: Path = Field(default_factory=lambda: BASE_DIR / "data" / "media" / "catalog.json")
    cors_origins: str | list[str] = Field(
        default_factory=lambda: ["http://localhost:8080", "http://127.0.0.1:8080"]
    )
    remote_model_timeout_seconds: float = Field(default=12.0, gt=0, le=120)
    model_meshgraphnet_url: str = "http://localhost:9001"
    model_fno_url: str = "http://localhost:9002"
    model_pinn_url: str = "http://localhost:9003"
    model_meshgraphnet_predict_path: str = "/predict"
    model_fno_predict_path: str = "/predict"
    model_pinn_predict_path: str = "/predict"

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return str(value).upper()

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("API_PREFIX must start with '/'.")
        if len(value) > 1 and value.endswith("/"):
            raise ValueError("API_PREFIX must not end with '/'.")
        return value

    @field_validator("media_catalog_path")
    @classmethod
    def resolve_media_catalog_path(cls, value: Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if value is None:
            return ["http://localhost:8080", "http://127.0.0.1:8080"]
        if isinstance(value, str):
            origins = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            origins = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ValueError("CORS_ORIGINS must be a comma-separated string or list.")
        if not origins:
            raise ValueError("CORS_ORIGINS must contain at least one origin.")
        return origins

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if origin == "*":
                continue
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("CORS_ORIGINS entries must be valid http(s) origins or '*'.")
        return value

    @field_validator("model_meshgraphnet_url", "model_fno_url", "model_pinn_url")
    @classmethod
    def validate_model_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Model service URLs must be valid http(s) URLs.")
        return value.rstrip("/")

    @field_validator(
        "model_meshgraphnet_predict_path",
        "model_fno_predict_path",
        "model_pinn_predict_path",
    )
    @classmethod
    def validate_predict_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("Model predict paths must start with '/'.")
        if "://" in value:
            raise ValueError("Model predict paths must be relative paths, not full URLs.")
        return value

    @model_validator(mode="after")
    def validate_environment_cors(self) -> "Settings":
        if self.environment == "production" and "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin is not allowed in production.")
        return self


def get_settings() -> Settings:
    return Settings()
