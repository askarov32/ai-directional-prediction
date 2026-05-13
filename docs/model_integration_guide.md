# New Model Integration Guide

This guide explains how to add a new remote model service to the thermoelastic direction prediction project.

Use it when you want to add a model route such as:

- `transformer`
- `lstm`
- `graph_transformer`
- any other service that exposes a compatible HTTP prediction API

The backend must remain an orchestrator. It should not implement the model internals. A new model is added as a remote service plus a backend client/adaptor.

## 1. How The Current Routing Works

The public frontend/backend request is always:

```text
POST /api/v1/predictions
```

The user selects:

```json
{
  "model": "meshgraphnet"
}
```

The backend then:

1. validates the request;
2. resolves `medium_id` from `backend/data/media/catalog.json`;
3. merges rock properties into the request;
4. chooses the correct model client by `model`;
5. calls the configured remote service endpoint;
6. normalizes the model response for the frontend.

The routing code is centered around:

```text
backend/app/domain/enums/model_type.py
backend/app/domain/services/prediction_router.py
backend/app/infrastructure/clients/
backend/app/api/dependencies.py
```

## 2. Required Model Service Contract

Every model service should expose:

```text
GET /health
GET /ready
POST /predict
```

`GET /health` can be simple:

```json
{
  "status": "ok",
  "service": "transformer"
}
```

`GET /ready` should tell the backend whether the service can accept predictions:

```json
{
  "status": "ready",
  "service": "transformer",
  "ready": true,
  "model_version": "transformer-v1"
}
```

`POST /predict` receives an enriched backend payload. The exact model can ignore fields it does not need, but the payload shape is:

```json
{
  "medium": {
    "id": "granite",
    "name": "Granite",
    "category": "igneous",
    "properties": {
      "rho": 2650.0,
      "porosity_total": 0.015,
      "porosity_effective": 0.005,
      "vp": 5.95,
      "vs": 3.4,
      "thermal_conductivity": 2.7,
      "heat_capacity": 790.0,
      "thermal_expansion": 0.0000075
    },
    "ranges": {
      "temperature_c": [-20.0, 500.0],
      "pressure_mpa": [0.1, 2000.0]
    },
    "metadata": {
      "source": "mvp_catalog_extension",
      "notes": "Starter preset."
    }
  },
  "scenario": {
    "temperature_c": 120.0,
    "pressure_mpa": 35.0,
    "time_ms": 12.0
  },
  "source": {
    "type": "thermal_pulse",
    "x": 0.15,
    "y": 0.4,
    "z": 0.0,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.0, 0.0]
  },
  "probe": {
    "x": 0.7,
    "y": 0.55,
    "z": 0.0
  },
  "domain": {
    "type": "rect_2d",
    "size": {
      "lx": 1.0,
      "ly": 1.0,
      "lz": 0.0
    },
    "resolution": {
      "nx": 128,
      "ny": 128,
      "nz": 1
    },
    "boundary_conditions": {
      "left": "fixed",
      "right": "free",
      "top": "insulated",
      "bottom": "insulated"
    }
  },
  "representation": "sequence",
  "routing_hint": "transformer"
}
```

The remote response must be normalizable by `backend/app/infrastructure/adapters/response_normalizer.py`.

Minimum flat response:

```json
{
  "direction_vector": [0.82, 0.57, 0.0],
  "azimuth_deg": 34.7,
  "elevation_deg": 0.0,
  "magnitude": 1.0,
  "wave_type": "attention_wavefront",
  "travel_time_ms": 11.8,
  "max_displacement": 0.0032,
  "max_temperature_perturbation": 1.7,
  "model_version": "transformer-v1"
}
```

The backend will return this normalized frontend response:

```json
{
  "model": "transformer",
  "medium": {
    "id": "granite",
    "name": "Granite",
    "category": "igneous"
  },
  "prediction": {
    "direction_vector": [0.82, 0.57, 0.0],
    "azimuth_deg": 34.7,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "attention_wavefront",
    "travel_time_ms": 11.8
  },
  "field_summary": {
    "max_displacement": 0.0032,
    "max_temperature_perturbation": 1.7
  },
  "meta": {
    "model_version": "transformer-v1",
    "latency_ms": 48,
    "request_id": "uuid"
  }
}
```

## 3. Backend Files To Change

For examples below, assume the new model id is:

```text
transformer
```

### 3.1 Add The Model Enum

Edit:

```text
backend/app/domain/enums/model_type.py
```

Add:

```python
class ModelType(str, Enum):
    MESHGRAPHNET = "meshgraphnet"
    FNO = "fno"
    PINN = "pinn"
    TRANSFORMER = "transformer"

    @property
    def label(self) -> str:
        return {
            ModelType.MESHGRAPHNET: "MeshGraphNet",
            ModelType.FNO: "FNO",
            ModelType.PINN: "PINN",
            ModelType.TRANSFORMER: "Transformer",
        }[self]
```

This automatically allows Pydantic to accept `"transformer"` in the public request contract.

### 3.2 Add Config Variables

Edit:

```text
backend/app/core/config.py
```

Add settings:

```python
model_transformer_url: str = "http://localhost:9004"
model_transformer_predict_path: str = "/predict"
```

Add `model_transformer_url` to the URL validator:

```python
@field_validator("model_meshgraphnet_url", "model_fno_url", "model_pinn_url", "model_transformer_url")
```

Add `model_transformer_predict_path` to the path validator:

```python
@field_validator(
    "model_meshgraphnet_predict_path",
    "model_fno_predict_path",
    "model_pinn_predict_path",
    "model_transformer_predict_path",
)
```

### 3.3 Add The Model Client

Create:

```text
backend/app/infrastructure/clients/transformer_client.py
```

Example:

```python
from __future__ import annotations

from typing import Any

from app.domain.entities.prediction import EnrichedPredictionRequest
from app.domain.enums.model_type import ModelType
from app.infrastructure.clients.base import BaseModelClient


class TransformerClient(BaseModelClient):
    def __init__(self, base_url: str, predict_path: str, timeout_seconds: float) -> None:
        super().__init__(
            model_type=ModelType.TRANSFORMER,
            service_name="Transformer",
            base_url=base_url,
            predict_path=predict_path,
            timeout_seconds=timeout_seconds,
        )

    def build_payload(self, request: EnrichedPredictionRequest) -> dict[str, Any]:
        payload = request.to_shared_payload()
        payload["representation"] = "sequence"
        payload["routing_hint"] = "transformer"
        return payload
```

Recommended `representation` values:

```text
meshgraphnet -> graph
fno -> grid
pinn -> physics_informed
transformer -> sequence
```

### 3.4 Register The Client In Dependency Injection

Edit:

```text
backend/app/api/dependencies.py
```

Import:

```python
from app.infrastructure.clients.transformer_client import TransformerClient
```

Add to `clients = [...]`:

```python
TransformerClient(
    base_url=settings.model_transformer_url,
    predict_path=settings.model_transformer_predict_path,
    timeout_seconds=settings.remote_model_timeout_seconds,
),
```

After this, `PredictionRouter` can route requests with `"model": "transformer"`.

## 4. Environment And Docker

### 4.1 Add `.env.example` Variables

Edit:

```text
.env.example
```

Add:

```env
MOCK_TRANSFORMER_PORT=9004
MODEL_TRANSFORMER_URL=http://mock-transformer:9000
MODEL_TRANSFORMER_PREDICT_PATH=/predict
```

If the model is a real external service, use:

```env
MODEL_TRANSFORMER_URL=http://your-transformer-host:8000
MODEL_TRANSFORMER_PREDICT_PATH=/predict
```

### 4.2 Add Backend Environment Variables In Compose

Edit:

```text
docker-compose.yml
```

Under `backend.environment`, add:

```yaml
MODEL_TRANSFORMER_URL: ${MODEL_TRANSFORMER_URL:-http://mock-transformer:9000}
MODEL_TRANSFORMER_PREDICT_PATH: ${MODEL_TRANSFORMER_PREDICT_PATH:-/predict}
```

If the model service is part of local compose, also add dependency:

```yaml
depends_on:
  mock-transformer:
    condition: service_healthy
```

### 4.3 Add A Mock Service For Local Demo

If the real model is not ready yet, reuse the existing mock service image:

```yaml
mock-transformer:
  build:
    context: ./mock-services
  container_name: mock-transformer
  environment:
    SERVICE_KIND: transformer
  ports:
    - "${MOCK_TRANSFORMER_PORT:-9004}:9000"
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/ready')"]
    interval: 15s
    timeout: 5s
    retries: 5
    start_period: 5s
```

Then edit:

```text
mock-services/common/predictor.py
```

Add a synthetic offset and wave type:

```python
SERVICE_OFFSETS = {
    "meshgraphnet": 0.12,
    "fno": -0.08,
    "transformer": 0.04,
}
```

```python
wave_type = {
    "meshgraphnet": "dominant_p",
    "fno": "coupled_field",
    "transformer": "attention_wavefront",
}.get(service_kind, "dominant_p")
```

## 5. Frontend Notes

The frontend loads model options from:

```text
GET /api/v1/models
```

So usually no frontend logic change is required. Once the backend returns:

```json
{
  "id": "transformer",
  "name": "Transformer",
  "status": "configured"
}
```

the selector should show it automatically.

Optional frontend polish:

```text
frontend/index.html
frontend/assets/styles/components.css
frontend/assets/scripts/ui.js
```

Typical optional changes:

- update static copy that lists available models;
- add a model badge color for `[data-model="transformer"]`;
- add a short display label if the UI has hardcoded model descriptions.

Do not change the public request payload unless the model requires a new user input.

## 6. Analytics And Comparison Scripts

If the repo has model comparison scripts, update their model list.

Typical file:

```text
analytics/scripts/generate_prediction_scenarios.py
```

Example:

```python
MODELS = ["meshgraphnet", "fno", "pinn", "transformer"]
```

If chart colors are hardcoded, add a color:

```python
MODEL_COLORS = {
    "meshgraphnet": "#60a5fa",
    "fno": "#34d399",
    "pinn": "#f59e0b",
    "transformer": "#a78bfa",
}
```

Then regenerate scenarios, predictions, and charts:

```bash
python3 analytics/scripts/generate_prediction_scenarios.py
python3 analytics/scripts/run_model_comparison_predictions.py --backend-url http://127.0.0.1:8000
python3 analytics/scripts/generate_model_comparison_charts.py
python3 analytics/scripts/generate_evaluation_summary.py
```

If the analytics scripts are not present in your branch, skip this step.

## 7. Tests To Update Or Add

Recommended minimum checks:

```text
backend/tests/test_api_smoke.py
backend/tests/test_predict_direction_use_case.py
backend/tests/test_model_client_errors.py
```

What to verify:

- `GET /api/v1/models` includes the new model id;
- request validation accepts `"model": "transformer"`;
- `PredictionRouter` can route to the new client;
- unavailable/timeout/malformed-response errors still return the existing JSON error shape.

Example assertion:

```python
assert {"meshgraphnet", "fno", "pinn", "transformer"} <= {item["id"] for item in response.json()}
```

## 8. Verification Commands

After changing code:

```bash
docker compose config --quiet
```

Start the stack:

```bash
docker compose up --build
```

Check backend model registry:

```bash
curl -s http://127.0.0.1:8000/api/v1/models
```

Expected:

```json
[
  {"id": "meshgraphnet", "name": "MeshGraphNet", "status": "configured"},
  {"id": "fno", "name": "FNO", "status": "configured"},
  {"id": "pinn", "name": "PINN", "status": "configured"},
  {"id": "transformer", "name": "Transformer", "status": "configured"}
]
```

Check readiness:

```bash
curl -s http://127.0.0.1:8000/api/v1/ready
curl -s http://127.0.0.1:9004/ready
```

Run a prediction:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "transformer",
    "medium_id": "granite",
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.4,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.7,
      "y": 0.55,
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    }
  }'
```

## 9. Friend-Friendly Checklist

For a new model named `transformer`, change:

```text
backend/app/domain/enums/model_type.py
backend/app/core/config.py
backend/app/infrastructure/clients/transformer_client.py
backend/app/api/dependencies.py
.env.example
docker-compose.yml
mock-services/common/predictor.py
README.md
docs/model_card.md
docs/demo_limitations.md
docs/ai_thermoelastic_architecture.md
```

Optional:

```text
frontend/index.html
frontend/assets/styles/components.css
analytics/scripts/*
backend/tests/*
```

The absolute minimum for a real hosted model:

1. Add `ModelType`.
2. Add config URL/path.
3. Add a backend client.
4. Register client in `dependencies.py`.
5. Set `MODEL_<MODEL>_URL`.
6. Make sure the remote service returns the required response fields.

## 10. Common Mistakes

- Adding a Docker service but forgetting to register the backend client.
- Adding a backend client but forgetting `ModelType`, so Pydantic rejects the request.
- Returning nested model output that the normalizer cannot parse.
- Forgetting `GET /ready`, causing `/api/v1/ready` to report the service as unavailable.
- Using a full URL as predict path. Predict paths must be relative, for example `/predict`.
- Changing frontend request shape for one model. The backend contract should stay unified.

