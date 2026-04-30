from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from pinn_service.inference_config import get_inference_config
from pinn_service.inference_service import CheckpointNotReadyError, PINNInferenceService
from pinn_service.service_schemas import PINNPredictionRequest


config = get_inference_config()
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
service = PINNInferenceService(config)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.try_initialize()
    yield


app = FastAPI(title="PINN Inference Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return service.health_payload()


@app.post("/predict")
async def predict(request: PINNPredictionRequest) -> dict:
    try:
        return service.predict(request)
    except CheckpointNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CHECKPOINT_NOT_READY",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("pinn_service.service").exception("Unhandled inference failure", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_FAILURE",
                "message": "PINN inference failed",
            },
        ) from exc
