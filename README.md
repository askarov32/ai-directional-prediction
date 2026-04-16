# Thermoelastic Direction Predictor MVP

Polished MVP thesis-demo for **“AI directional prediction of the propagation of thermoelastic waves in geological media”**.

The project ships a full local stack:

- `FastAPI` backend orchestrator
- native `HTML/CSS/JavaScript` frontend
- mock model services for `MeshGraphNet`, `FNO`, and `PINN`
- `Docker Compose` startup for quick local demos

The backend does **not** implement ML models. It acts as an orchestration layer that validates a unified request, resolves the geological medium, merges scenario inputs with rock presets, routes the request to the correct remote model service, and normalizes the response.

## Overview

Main user flow:

1. Choose a geological medium from the JSON preset catalog
2. Inspect physical properties
3. Configure the thermoelastic scenario, source, probe, and domain
4. Select the model route: `meshgraphnet`, `fno`, or `pinn`
5. Run prediction
6. Inspect normalized direction metrics and the 2D visualization

## Architecture Summary

### Backend

- `FastAPI` REST API under `/api/v1`
- domain use case `PredictDirectionUseCase`
- JSON-based media repository
- dedicated `PredictionRouter`
- separate model clients for `MeshGraphNet`, `FNO`, `PINN`
- response normalization layer
- consistent JSON error payloads

### Frontend

- Vanilla HTML, CSS, and JavaScript only
- responsive glassmorphism-style scientific UI
- SVG-based domain visualization
- dynamic media/model loading from backend
- inline validation and debug panel

### Mock Services

- lightweight FastAPI services used by default in Docker
- deterministic synthetic outputs with slightly different behavior per model type
- easy to replace with real model hosts

## Project Structure

```text
.
├── README.md
├── docker-compose.yml
├── .env.example
├── backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app
│   │   ├── main.py
│   │   ├── api
│   │   │   ├── dependencies.py
│   │   │   └── routes
│   │   │       ├── health.py
│   │   │       ├── media.py
│   │   │       ├── models.py
│   │   │       └── predictions.py
│   │   ├── core
│   │   │   ├── config.py
│   │   │   ├── exceptions.py
│   │   │   └── logging.py
│   │   ├── domain
│   │   │   ├── entities
│   │   │   │   ├── medium.py
│   │   │   │   └── prediction.py
│   │   │   ├── enums
│   │   │   │   └── model_type.py
│   │   │   ├── services
│   │   │   │   ├── medium_catalog.py
│   │   │   │   └── prediction_router.py
│   │   │   └── use_cases
│   │   │       └── predict_direction.py
│   │   ├── infrastructure
│   │   │   ├── adapters
│   │   │   │   └── response_normalizer.py
│   │   │   ├── clients
│   │   │   │   ├── base.py
│   │   │   │   ├── fno_client.py
│   │   │   │   ├── meshgraphnet_client.py
│   │   │   │   └── pinn_client.py
│   │   │   └── repositories
│   │   │       └── media_repository.py
│   │   └── schemas
│   │       ├── media.py
│   │       └── prediction.py
│   └── data
│       └── media
│           └── catalog.json
├── frontend
│   ├── Dockerfile
│   ├── index.html
│   ├── nginx.conf
│   └── assets
│       ├── icons
│       │   └── wave-grid.svg
│       ├── scripts
│       │   ├── api.js
│       │   ├── app.js
│       │   ├── charts.js
│       │   ├── form.js
│       │   ├── state.js
│       │   ├── ui.js
│       │   └── validators.js
│       └── styles
│           ├── animations.css
│           ├── base.css
│           ├── components.css
│           ├── layout.css
│           ├── reset.css
│           └── variables.css
└── mock-services
    ├── Dockerfile
    ├── main.py
    ├── requirements.txt
    └── common
        └── predictor.py
```

## Environment Variables

Copy `.env.example` to `.env` if you want to customize ports or remote endpoints:

```bash
cp .env.example .env
```

Important variables:

- `BACKEND_PORT`
- `FRONTEND_PORT`
- `MOCK_MESHGRAPHNET_PORT`
- `MOCK_FNO_PORT`
- `MOCK_PINN_PORT`
- `MODEL_MESHGRAPHNET_URL`
- `MODEL_FNO_URL`
- `MODEL_PINN_URL`
- `MODEL_MESHGRAPHNET_PREDICT_PATH`
- `MODEL_FNO_PREDICT_PATH`
- `MODEL_PINN_PREDICT_PATH`
- `REMOTE_MODEL_TIMEOUT_SECONDS`
- `CORS_ORIGINS`

Default Docker routing:

- backend: `http://localhost:8000`
- frontend: `http://localhost:8080`
- mock MeshGraphNet: `http://localhost:9001`
- mock FNO: `http://localhost:9002`
- mock PINN: `http://localhost:9003`

## Quick Start

Run the whole stack:

```bash
docker compose up --build
```

Then open:

- Frontend: [http://localhost:8080](http://localhost:8080)
- Backend OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

All API routes are versioned under `/api/v1`.

- `GET /api/v1/health`
- `GET /api/v1/media`
- `GET /api/v1/media/{medium_id}`
- `GET /api/v1/models`
- `POST /api/v1/predictions`

### Unified Prediction Request

Example payload:

```json
{
  "model": "meshgraphnet",
  "medium_id": "sandstone_medium",
  "scenario": {
    "temperature_c": 120.0,
    "pressure_mpa": 35.0,
    "time_ms": 12.0
  },
  "source": {
    "type": "thermal_pulse",
    "x": 0.15,
    "y": 0.40,
    "z": 0.0,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.0, 0.0]
  },
  "probe": {
    "x": 0.70,
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
}
```

### Unified Prediction Response

```json
{
  "model": "meshgraphnet",
  "medium": {
    "id": "sandstone_medium",
    "name": "Sandstone (medium)",
    "category": "sedimentary"
  },
  "prediction": {
    "direction_vector": [0.82, 0.57, 0.0],
    "azimuth_deg": 34.7,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "dominant_p",
    "travel_time_ms": 11.8
  },
  "field_summary": {
    "max_displacement": 0.0032,
    "max_temperature_perturbation": 1.7
  },
  "meta": {
    "model_version": "mock-meshgraphnet-v1",
    "latency_ms": 48,
    "request_id": "uuid"
  }
}
```

## How Model Routing Works

The routing logic lives in the backend domain layer:

1. The request is validated by Pydantic schemas
2. `PredictDirectionUseCase` resolves the medium from the JSON catalog
3. Scenario values are checked against medium ranges
4. The backend enriches the request with preset rock properties
5. `PredictionRouter` selects the correct client by `model`
6. The selected client builds a model-specific payload
7. The backend calls the configured remote model endpoint
8. `ResponseNormalizer` converts the model response to a frontend-friendly format

Internal payload representations:

- `meshgraphnet` -> `representation: "graph"`
- `fno` -> `representation: "grid"`
- `pinn` -> `representation: "physics_informed"`

## Replacing Mock Services With Real Models

By default, Docker uses the included mock services. To switch to real model hosts:

1. Edit `.env`
2. Point these variables to your real services:

```bash
MODEL_MESHGRAPHNET_URL=http://your-meshgraphnet-host:8001
MODEL_FNO_URL=http://your-fno-host:8002
MODEL_PINN_URL=http://your-pinn-host:8003
```

3. If the real services use a different route than `/predict`, update:

```bash
MODEL_MESHGRAPHNET_PREDICT_PATH=/your-endpoint
MODEL_FNO_PREDICT_PATH=/your-endpoint
MODEL_PINN_PREDICT_PATH=/your-endpoint
```

No backend code changes are required for that swap.

## Running Backend Separately

From the `backend` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you do this outside Docker, make sure the model URLs point to reachable hosts such as:

```bash
export MODEL_MESHGRAPHNET_URL=http://localhost:9001
export MODEL_FNO_URL=http://localhost:9002
export MODEL_PINN_URL=http://localhost:9003
```

## Running Frontend Separately

The frontend is plain static assets. From the `frontend` directory:

```bash
python -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080).

The frontend expects the backend at `http://localhost:8000/api/v1` by default.

## Error Handling

The backend returns consistent JSON errors:

```json
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "MeshGraphNet service is unavailable",
    "details": {
      "url": "http://mock-meshgraphnet:9000/predict"
    }
  }
}
```

Implemented error categories include:

- unknown medium
- unsupported model
- temperature out of range
- pressure out of range
- invalid coordinates
- invalid resolution
- remote model unavailable
- remote model timeout
- malformed remote model response

## Screenshots / Demo Notes

Suggested placeholders for your thesis report:

- main hero and form workspace
- prediction result card with azimuth/elevation
- SVG domain visualization with source/probe/arrow
- debug panel showing normalized JSON response

## Notes

- The project is intentionally 2D-first for MVP clarity
- Domain structure is already ready for future 3D expansion
- Mock services are synthetic and deterministic, not scientific solvers
- The medium catalog is loaded from JSON, not hardcoded in route handlers
