from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_default_cors_is_localhost_only(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    settings = Settings()

    assert settings.cors_origins == ["http://localhost:8080", "http://127.0.0.1:8080"]


def test_settings_parses_csv_cors_origins():
    settings = Settings(cors_origins="http://localhost:8080,http://127.0.0.1:8080")

    assert settings.cors_origins == ["http://localhost:8080", "http://127.0.0.1:8080"]


def test_settings_parses_csv_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")

    settings = Settings()

    assert settings.cors_origins == ["http://localhost:8080", "http://127.0.0.1:8080"]


def test_settings_rejects_production_wildcard_cors():
    with pytest.raises(ValidationError, match="Wildcard CORS origin is not allowed in production"):
        Settings(environment="production", cors_origins="*")


def test_settings_rejects_invalid_api_prefix():
    with pytest.raises(ValidationError, match="API_PREFIX must start"):
        Settings(api_prefix="api/v1")


def test_settings_rejects_invalid_model_url():
    with pytest.raises(ValidationError, match="Model service URLs must be valid"):
        Settings(model_pinn_url="pinn-service:9000")


def test_settings_rejects_invalid_timeout():
    with pytest.raises(ValidationError):
        Settings(remote_model_timeout_seconds=0)
