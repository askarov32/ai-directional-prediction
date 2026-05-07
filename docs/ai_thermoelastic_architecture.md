# AI Thermoelastic Prediction Architecture

This document describes the current MVP implementation, not an aspirational API.

## Service Map

The local Docker stack contains:

- `frontend`: nginx-served vanilla HTML/CSS/JavaScript UI on `localhost:8080`
- `backend`: FastAPI orchestration/API gateway on `localhost:8000`
- `mock-meshgraphnet`: synthetic MeshGraphNet-compatible service on `localhost:9001`
- `mock-fno`: synthetic FNO-compatible service on `localhost:9002`
- `pinn-service`: checkpoint-based PINN inference service on `localhost:9003`

The frontend calls backend through nginx at `/api/v1`. The backend calls model services through Docker service names:

- `http://mock-meshgraphnet:9000/predict`
- `http://mock-fno:9000/predict`
- `http://pinn-service:9000/predict`

## Backend Layers

The backend is organized as a small clean architecture:

- `api/routes`: HTTP routes and request/response schema binding
- `schemas`: Pydantic API contracts
- `domain/entities`: medium and prediction entities
- `domain/services`: catalog and model routing logic
- `domain/use_cases`: `PredictDirectionUseCase`
- `domain/ports`: protocols used by domain/application code
- `infrastructure/clients`: HTTP clients for model services
- `infrastructure/repositories`: JSON media catalog repository
- `infrastructure/adapters`: remote response validation and normalization

Dependency direction is:

```text
api -> use_cases/domain services -> domain ports
infrastructure -> implements domain ports
```

## Actual API Endpoints

All backend endpoints are under `/api/v1`.

- `GET /api/v1/health`: liveness
- `GET /api/v1/ready`: readiness for catalog and model services
- `GET /api/v1/media`: geological medium catalog
- `GET /api/v1/media/{medium_id}`: one medium preset
- `GET /api/v1/models`: configured model routes
- `POST /api/v1/predictions`: unified prediction request

Model services expose:

- `GET /health`
- `GET /ready`
- `POST /predict`

## Geological Media Storage

The current catalog lives at:

```text
backend/data/media/catalog.json
```

Each medium contains:

- stable `id`
- display `name`
- `category`
- physical `properties`
- valid scenario `ranges`
- metadata notes/source

The backend repository loads this JSON file and exposes it through `GET /api/v1/media`.

## Unified Prediction Contract

The frontend sends one canonical payload to:

```text
POST /api/v1/predictions
```

Top-level fields:

- `model`: `meshgraphnet`, `fno`, or `pinn`
- `medium_id`
- `scenario`
- `source`
- `probe`
- `domain`

The backend:

1. validates the payload;
2. resolves the selected medium;
3. checks temperature/pressure against medium ranges;
4. merges medium properties into the model request;
5. routes by `model`;
6. calls the configured service;
7. validates the remote model response;
8. normalizes it into the frontend response.

## Model Routing

Routing is implemented in `PredictionRouter`.

- `meshgraphnet` routes to `MeshGraphNetClient`
- `fno` routes to `FNOClient`
- `pinn` routes to `PINNClient`

The model-specific adapters currently set:

- MeshGraphNet: `representation = "graph"`
- FNO: `representation = "grid"`
- PINN: `representation = "physics_informed"`, `routing_hint = "pinn"`

## Model Service Status

`GET /api/v1/models` returns configured model routes.

`GET /api/v1/ready` actively checks model service readiness. For the PINN service, readiness means:

- checkpoint path resolved;
- checkpoint loaded;
- deterministic smoke inference passed;
- output shape and finite values verified.

## PINN Implementation Notes

The current PINN service is a pragmatic hybrid baseline:

- checkpoint model input features: `x, y, z, t, E, nu, rho, alpha, k, Cp`
- neural outputs: `temperature_k`, `disp_x`, `disp_y`, `disp_z`
- final API prediction: neural outputs plus geometry/material postprocessing

The service returns extra diagnostics:

- `model_outputs`
- `postprocessed_prediction`
- `diagnostics`

The backend normalized response remains stable and ignores extra fields.

## Docker Architecture

The Compose stack is demo-oriented:

- backend waits for model services to be healthy;
- frontend waits for backend readiness;
- readiness endpoints are used by healthchecks;
- Python service containers run as non-root `appuser`;
- nginx serves static assets and proxies `/api/` to backend.

Run:

```bash
docker compose up --build
```

Smoke checks:

```bash
docker compose config --quiet
curl -s http://localhost:8000/api/v1/ready
curl -s http://localhost:9003/ready
curl -s http://localhost:8080/api/v1/models
```

## What Is Mock vs Real

- MeshGraphNet is mocked by `mock-services`.
- FNO is mocked by `mock-services`.
- PINN uses a real PyTorch checkpoint when available.
- The current PINN prediction is not a full real-time PDE solver; it is a hybrid neural + physics-informed + postprocessed MVP baseline.

For a thesis/demo-safe statement, use:

> The application demonstrates an extensible orchestration layer and a checkpoint-based PINN baseline for directional thermoelastic-wave prediction. MeshGraphNet and FNO services are currently represented by deterministic mock services unless replaced by real model hosts.

## Scientific Minimum Inputs

Temperature alone is not enough for directional propagation. The current MVP uses:

- medium physical properties;
- temperature;
- pressure;
- source type, position, amplitude, frequency, direction;
- probe coordinates;
- domain size, resolution, and boundary conditions;
- observation time.

Future scientific improvements should add:

- anisotropy tensors;
- heterogeneous spatial material fields;
- fluid saturation;
- source time functions;
- validated train/validation/test split;
- independent experimental or high-fidelity numerical validation.
