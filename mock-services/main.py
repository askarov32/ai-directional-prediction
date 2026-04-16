from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from common.predictor import generate_prediction


class GenericPayload(BaseModel):
    medium: dict
    scenario: dict
    source: dict
    probe: dict
    domain: dict
    representation: str


service_kind = os.getenv("SERVICE_KIND", "meshgraphnet")
app = FastAPI(title=f"Mock {service_kind.title()} Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": service_kind}


@app.post("/predict")
async def predict(payload: GenericPayload) -> dict:
    return generate_prediction(service_kind, payload.model_dump())
