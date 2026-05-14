from __future__ import annotations

import logging

from fastapi import FastAPI

from .routes import router
from ..inference.predictor import FNOInferenceService
from ..utils.config import FNOServiceConfig, get_service_config


def create_app(config: FNOServiceConfig | None = None) -> FastAPI:
    resolved_config = config or get_service_config()
    logging.basicConfig(
        level=getattr(logging, resolved_config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = FastAPI(title="FNO Inference Service", version="0.1.0")
    app.state.fno_service = FNOInferenceService(resolved_config)
    app.include_router(router)
    return app


app = create_app()
