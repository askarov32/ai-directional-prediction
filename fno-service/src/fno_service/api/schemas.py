from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    medium: dict[str, Any]
    scenario: dict[str, Any]
    source: dict[str, Any]
    probe: dict[str, Any]
    domain: dict[str, Any]
    representation: str = "grid"
    routing_hint: str | None = None
