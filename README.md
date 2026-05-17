# Thermoelastic Direction Predictor MVP

Polished MVP thesis-demo for **“AI directional prediction of the propagation of thermoelastic waves in geological media”**.

The project ships a full local stack:

- `FastAPI` backend orchestrator
- native `HTML/CSS/JavaScript` frontend
- integrated `MeshGraphNet` service with demo fallback mode
- dedicated `FNO` service with checkpoint-based `FNO2d` inference and fallback mode
- a dedicated checkpoint-based `PINN` service
- `Docker Compose` startup for quick local demos

The backend does **not** implement ML models. It acts as an orchestration layer that validates a unified request, resolves the geological medium, merges scenario inputs with rock presets, routes the request to the correct remote model service, and normalizes the response.

## Overview

Main user flow:

1. Choose a geological medium from the JSON preset catalog
2. Inspect physical properties
3. Configure the planar source, probe, and observation time
4. Select the model route: `meshgraphnet`, `fno`, `transformer`, or `pinn`
5. Run prediction
6. Inspect the normalized prediction response and 2D visualization

## Architecture Summary

### Backend

- `FastAPI` REST API under `/api/v1`
- domain use case `PredictDirectionUseCase`
- JSON-based media repository
- dedicated `PredictionRouter`
- separate model clients for `MeshGraphNet`, `FNO`, `Transformer`, `PINN`
- response normalization layer
- consistent JSON error payloads

### Frontend

- Vanilla HTML, CSS, and JavaScript only
- responsive minimalist dark scientific UI
- SVG-based domain visualization
- dynamic media/model loading from backend
- typed API errors, inline validation, and debug panel

### Model Services

- integrated FastAPI `MeshGraphNet` service under `mgn-service`
- dedicated FastAPI `FNO` service under `fno-service`
- a checkpoint-based `PINN` inference service with readiness diagnostics
- easy to replace each service host independently

For scientific scope and known limitations, see:

- [Model Card](docs/model_card.md)
- [Demo Limitations](docs/demo_limitations.md)
- [Frontend API v2 Usage](docs/frontend_api_v2_usage.md)
- [New Model Integration Guide](docs/model_integration_guide.md)
- [FNO Integration Roadmap](docs/FNO_INTEGRATION_ROADMAP.md)
- [FNO Phase 1 Audit](docs/FNO_PHASE1_AUDIT.md)
- [FNO Service Guide](docs/FNO_SERVICE.md)
- [FNO Training Guide](docs/FNO_TRAINING.md)
- [Windows FNO Training Guide](docs/windows_fno_training.md)
- [FNO API Contract](docs/FNO_API_CONTRACT.md)
- [FNO Dataset Format](docs/FNO_DATASET_FORMAT.md)
- [PINN Architecture Details](docs/pinn_architecture_details.md)
- [Windows PINN Training Guide](docs/windows_pinn_training.md)
- [Physics Theory Notes](physics-theory/README.md)

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
│   │   │   │   ├── pinn_client.py
│   │   │   │   └── transformer_client.py
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
├── mock-services
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── common
│       └── predictor.py
├── mgn-service
│   ├── Dockerfile
│   ├── README.md
│   ├── configs
│   ├── scripts
│   └── src
│       └── service
│           └── api.py
└── pinn-service
    ├── Dockerfile
    ├── README.md
    ├── requirements.txt
    └── src
        └── pinn_service
            ├── cli.py
            ├── comsol_parser.py
            ├── dataset_builder.py
            ├── inference_config.py
            ├── inference_service.py
            ├── inference_utils.py
            ├── losses.py
            ├── model.py
            ├── service_app.py
            ├── service_schemas.py
            ├── train.py
            ├── trainer.py
            ├── training_config.py
            └── training_data.py
```

## Environment Variables

Copy `.env.example` to `.env` if you want to customize ports or remote endpoints:

```bash
cp .env.example .env
```

Important variables:

- `BACKEND_PORT`
- `FRONTEND_PORT`
- `MGN_SERVICE_PORT`
- `FNO_SERVICE_PORT`
- `MOCK_FNO_PORT`
- `MOCK_TRANSFORMER_PORT`
- `PINN_SERVICE_PORT`
- `MODEL_MESHGRAPHNET_URL`
- `MODEL_FNO_URL`
- `MODEL_TRANSFORMER_URL`
- `MODEL_PINN_URL`
- `MODEL_MESHGRAPHNET_PREDICT_PATH`
- `MODEL_FNO_PREDICT_PATH`
- `MODEL_TRANSFORMER_PREDICT_PATH`
- `MODEL_PINN_PREDICT_PATH`
- `REMOTE_MODEL_TIMEOUT_SECONDS`
- `CORS_ORIGINS`
- `MGN_DATASET_ID`
- `MGN_CHECKPOINT_PATH`
- `MGN_DEVICE`
- `MGN_ALLOW_FALLBACK`
- `FNO_CHECKPOINT_PATH`
- `FNO_CONFIG_PATH`
- `FNO_DATASET_PATH`
- `FNO_DEVICE`
- `FNO_ALLOW_FALLBACK`
- `PINN_CHECKPOINT_PATH`
- `PINN_DEVICE`
- `PINN_TIME_SCALE`

Default Docker routing:

- backend: `http://localhost:8000`
- frontend: `http://localhost:8080`
- MeshGraphNet service: `http://localhost:9001`
- FNO service: `http://localhost:9002`
- mock FNO with `--profile demo-mocks`: `http://localhost:9012`
- mock Transformer: `http://localhost:9004`
- pinn-service: `http://localhost:9003`
- transformer-service: `http://localhost:9004`

## Quick Start

Run the whole stack:

```bash
docker compose up --build
```

The repository includes a baseline `PINN` checkpoint under `pinn-service/artifacts/checkpoints/baseline`, so the compose stack is intended to be demo-ready without a separate training step.

If you are upgrading from an older local setup that still had `mock-pinn`, run this once to clean old containers:

```bash
docker compose up -d --remove-orphans
```

Then open:

- Frontend: [http://localhost:8080](http://localhost:8080)
- Backend OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

Quick smoke checks:

```bash
docker compose config --quiet
curl -s http://localhost:8000/api/v1/ready
curl -s http://localhost:9002/ready
curl -s http://localhost:9003/ready
curl -s http://localhost:8080/api/v1/models
```

Sample prediction through the frontend nginx proxy:

```bash
curl -s -X POST http://localhost:8080/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "pinn",
    "medium_id": "sandstone_medium",
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

## API Endpoints

All API routes are versioned under `/api/v1`.

- `GET /api/v1/health`
- `GET /api/v1/ready`
- `GET /api/v1/media`
- `GET /api/v1/media/{medium_id}`
- `GET /api/v1/models`
- `POST /api/v1/predictions`

## Automated Checks

GitHub Actions runs a lightweight CI workflow on pull requests and pushes to `main`, `pinn`, and `codex/**` branches.

The workflow checks:

- backend tests;
- PINN contract tests;
- Docker Compose configuration syntax.

Local equivalents:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=pinn-service/src python -m pytest pinn-service/tests
docker compose config --quiet
```

CI intentionally does not train models or build the full Docker stack, because those steps are heavier and depend on model artifacts/hardware.

## Prediction Input And Output

The prediction screen is organized around one unified request, regardless of the selected model route.

User-facing inputs:

- `Geological medium`: a preset loaded from `backend/data/media/catalog.json`; the current demo catalog includes sandstone, limestone, basalt, and granite.
- `Model route`: `meshgraphnet`, `fno`, `transformer`, or `pinn`.
- `Source`: planar `x_m`, `y_m` coordinates inside the fixed `1 m x 1 m` domain.
- `Probe`: planar `x_m`, `y_m` coordinates inside the fixed `1 m x 1 m` domain.
- `Observation`: `time_s`.
- `Scenario`: a locked prototype template resolved as `thermal_source_type=point`, `mechanical_constraint=free`, `boundary_condition_type=prototype_simplified`.

The current frontend sends API Contract v2 only:

- `schema_version: "2.0"`
- `model`
- `medium_id`
- `geometry.dimension: 2`
- `geometry.source`
- `geometry.probe`
- `observation.time_s`
- `scenario.*`

The frontend does not expose or send:

- `z` coordinates
- `dimension: 3`
- custom domain size or resolution
- `frequency_hz`
- source amplitude or source direction
- old temperature/pressure form fields
- backend-owned thermal overrides such as `reference_temperature_k`

The frontend displays the normalized prediction response as:

- a 2D source-to-probe SVG geometry diagram
- separate result cards for thermal, displacement, directional, temporal, and model metadata blocks
- a diagnostics/fallback panel with warnings, notes, request id, and schema version
- an optional spatial heatmap only when `optional_outputs.field_grid` is returned
- a collapsible debug section with raw JSON for QA and service inspection

### Unified Prediction Request

Example payload:

```json
{
  "schema_version": "2.0",
  "model": "meshgraphnet",
  "medium_id": "sandstone",
  "geometry": {
    "dimension": 2,
    "source": {
      "x_m": 0.15,
      "y_m": 0.40
    },
    "probe": {
      "x_m": 0.70,
      "y_m": 0.55
    }
  },
  "observation": {
    "time_s": 0.012
  },
  "scenario": {
    "thermal_source_type": "point",
    "mechanical_constraint": "free",
    "boundary_condition_type": "prototype_simplified"
  }
}
```

### Unified Prediction Response

```json
{
  "schema_version": "2.0",
  "request_id": "uuid",
  "status": "ok",
  "model": {
    "name": "meshgraphnet",
    "version": "real-meshgraphnet-v1",
    "route": "/predict",
    "inference_time_ms": 48.0,
    "fallback_used": false,
    "fallback_reason": null
  },
  "material": {
    "id": "sandstone",
    "name": "Sandstone",
    "category": "sedimentary siliciclastic"
  },
  "geometry": {
    "dimension": 2,
    "source": { "x_m": 0.15, "y_m": 0.40 },
    "probe": { "x_m": 0.70, "y_m": 0.55 },
    "propagation_vector_m": { "dx": 0.55, "dy": 0.15 },
    "unit_direction": { "x": 0.9648, "y": 0.2631 },
    "distance_m": 0.570088,
    "azimuth_deg": 15.255119
  },
  "prediction": {
    "thermal": {
      "temperature_k": { "value": 293.2741, "source": "direct_model_prediction" },
      "temperature_perturbation_k": {
        "value": 3.2105,
        "reference_temperature_k": 273.15,
        "source": "direct_model_prediction"
      }
    },
    "displacement": {
      "components_m": { "u": -5.50e-05, "v": -4.76e-05 },
      "magnitude_m": 7.27e-05,
      "components_source": "direct_model_prediction",
      "magnitude_source": "derived_from_u_v"
    },
    "directional_response": {
      "distance_m": 0.570088,
      "azimuth_deg": 15.255119,
      "response_magnitude_score": 1.0
    },
    "temporal_response": {
      "travel_time_s": 0.002940175,
      "source": "direct_model_prediction"
    }
  },
  "optional_outputs": {
    "confidence_score": null,
    "field_summary": {
      "max_displacement_m": 0.002045799,
      "max_temperature_perturbation_k": 3.21051
    },
    "field_grid": null,
    "strain": null,
    "stress": null
  },
  "diagnostics": {
    "fallback_used": false,
    "fallback_reason": null,
    "warnings": [],
    "notes": ["Prototype prediction; not a field-validated thermoelastic simulation."]
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
- `transformer` -> `representation: "sequence"`
- `pinn` -> `representation: "physics_informed"`

## Replacing Model Services

By default, Docker uses:

- integrated `mgn-service` for `MeshGraphNet`;
- integrated `fno-service` for `FNO`;
- integrated checkpoint-based `pinn-service` for `PINN`;
- `mock-transformer` for `Transformer`.

`mgn-service` and `fno-service` can both run in fallback mode for local demos when artifacts are missing or still being prepared.

To switch to external model hosts:

1. Edit `.env`
2. Point these variables to your real services:

```bash
MODEL_MESHGRAPHNET_URL=http://your-meshgraphnet-host:8001
MODEL_FNO_URL=http://your-fno-host:8002
MODEL_TRANSFORMER_URL=http://your-transformer-host:8004
MODEL_PINN_URL=http://your-pinn-host:8003
```

3. If the real services use a different route than `/predict`, update:

```bash
MODEL_MESHGRAPHNET_PREDICT_PATH=/your-endpoint
MODEL_FNO_PREDICT_PATH=/your-endpoint
MODEL_TRANSFORMER_PREDICT_PATH=/your-endpoint
MODEL_PINN_PREDICT_PATH=/your-endpoint
```

No backend code changes are required for that swap.

## MeshGraphNet Service

The `MeshGraphNet` route is wired to `mgn-service`.

That service:

- exposes `GET /health`, `GET /ready`, and `POST /predict`;
- can run a real rollout when `MGN_DATASET_ID` and `MGN_CHECKPOINT_PATH` point to prepared artifacts;
- returns a valid fallback directional response when `MGN_ALLOW_FALLBACK=true` and artifacts are not present;
- keeps the same backend-normalized response contract as other model routes.

Useful variables:

```bash
MGN_DATASET_ID=sandstone_comsol_real
MGN_CHECKPOINT_PATH=outputs/checkpoints/best_model.pt
MGN_DEVICE=cuda
MGN_ALLOW_FALLBACK=true
```

## FNO Service

The `FNO` route is now wired to `fno-service`.

That service:

- exposes `GET /health`, `GET /ready`, and `POST /predict`;
- loads `best_model.pth` or `model.pth` from `FNO_CHECKPOINT_PATH`;
- loads its local regular-grid dataset from `FNO_DATASET_PATH`;
- combines the backend request with local grid conditioning;
- runs `FNO2d` inference for the current MVP baseline;
- falls back to a deterministic response when `FNO_ALLOW_FALLBACK=true` and no checkpoint is present.

Current FNO baseline limitations:

- current trained/inference path is `rect_2d` only;
- current regular grid must have `Z=1`;
- the request does not send full grid tensors through the backend;
- the service uses local dataset conditioning plus scenario/source/probe adjustments.

Useful variables:

```bash
FNO_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline
FNO_CONFIG_PATH=/app/configs/inference.yaml
FNO_DATASET_PATH=/app/artifacts/datasets/sandstone_fno
FNO_DEVICE=cpu
FNO_ALLOW_FALLBACK=true
```

CUDA note:

- set `FNO_DEVICE=cuda` on a machine with a CUDA-enabled PyTorch build;
- if `cuda` is requested but unavailable, the current service falls back to CPU instead of crashing.

To build a local demo baseline checkpoint and dataset in the default artifact paths:

```bash
./fno-service/train_baseline.sh
```

That helper:

- converts the demo PINN structured dataset into a regular FNO grid under `fno-service/artifacts/datasets/sandstone_fno`;
- trains a small `FNO2d` baseline;
- writes `best_model.pth` under `fno-service/artifacts/checkpoints/baseline`.

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
export MODEL_TRANSFORMER_URL=http://localhost:9004
export MODEL_PINN_URL=http://localhost:9003
```

## Running Frontend Separately

The frontend is plain static assets. From the `frontend` directory:

```bash
python -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080).

When served through Docker/nginx, the frontend uses `/api/v1`. If opened directly from `file://`, it falls back to `http://localhost:8000/api/v1`.

Current frontend scope:

- API Contract v2 only
- fixed 2D `1 m x 1 m` source-probe workflow
- no visible 3D controls in the UI
- optional spatial heatmap only when a model route returns `optional_outputs.field_grid`

For the current UI behavior and limitations, see [Frontend API v2 Usage](docs/frontend_api_v2_usage.md).

## PINN Service

The `PINN` route is now wired to a dedicated service instead of the old generic mock.

That service:

- auto-loads the best checkpoint from `/app/artifacts/checkpoints/baseline`
- derives `E` and `nu` from `rho`, `Vp`, and `Vs` during inference
- predicts `T, u, v, w`
- converts the result into direction and field summary metrics

If no checkpoint is present:

- `GET /health` on the PINN service reports `ready: false`
- `GET /ready` returns `503`
- `POST /predict` returns a clear `503` error instead of crashing

Current PINN responses are intentionally backward-compatible with the backend normalizer. The service also returns diagnostic fields such as `model_outputs`, `postprocessed_prediction`, and `diagnostics`; the backend ignores those extra fields for the normalized frontend response.

To refresh the baseline checkpoint locally:

```bash
./pinn-service/train_baseline.sh
```

That helper script writes a stronger CPU-friendly baseline to:

- `pinn-service/artifacts/checkpoints/baseline`
- and the inference service auto-prefers `best_model.pth` over `model.pth`

## Current PINN Training Objective

The current checkpoint is a hybrid supervised PINN baseline, not yet a full coupled thermoelastic PDE solver.

Training input:

```text
X = [x, y, z, t, E, nu, rho, alpha, k, Cp]
```

Primary neural-network output:

```text
Y = [T, u, v, w]
```

The total objective is:

```latex
\mathcal{L}_{total}
= \lambda_{sup}\mathcal{L}_{sup}
+ \lambda_{vel}\mathcal{L}_{vel}
+ \lambda_{temp}\mathcal{L}_{temp}
```

with the current default weights:

```latex
\lambda_{sup}=1.0,\qquad
\lambda_{vel}=0.25,\qquad
\lambda_{temp}=0.05
```

The supervised data loss is:

```latex
\mathcal{L}_{sup}
= \operatorname{MSE}
\left(
[\hat{T},\hat{u},\hat{v},\hat{w}],
[T,u,v,w]
\right)
```

The velocity consistency loss compares time derivatives of predicted displacement with COMSOL velocity targets:

```latex
\mathcal{L}_{vel}
= \operatorname{MSE}
\left(
\left[
\frac{\partial \hat{u}}{\partial t},
\frac{\partial \hat{v}}{\partial t},
\frac{\partial \hat{w}}{\partial t}
\right],
[u_t,v_t,w_t]
\right)
```

The thermal residual uses a diffusion-style heat equation:

```latex
a = \frac{k}{\rho C_p}
```

```latex
R_T =
\frac{\partial \hat{T}}{\partial t}
- a
\left(
\frac{\partial^2 \hat{T}}{\partial x^2}
+ \frac{\partial^2 \hat{T}}{\partial y^2}
+ \frac{\partial^2 \hat{T}}{\partial z^2}
\right)
```

```latex
\mathcal{L}_{temp} = \operatorname{mean}(R_T^2)
```

## Error Handling

The backend returns consistent JSON errors:

```json
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "MeshGraphNet service is unavailable",
    "details": {
      "url": "http://mgn-service:9000/predict"
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
- PINN output combines neural-network inference with geometry/material postprocessing
- The medium catalog is loaded from JSON, not hardcoded in route handlers

## PINN Dataset Preparation

The repository now includes a dedicated COMSOL CSV preprocessing step under [pinn-service/README.md](pinn-service/README.md).

Use it to turn the six COMSOL exports into:

- a structured field dataset for research and inspection
- an optional flattened training matrix for future PINN experiments

Quick example:

```bash
pip install -r pinn-service/requirements.txt
PYTHONPATH=pinn-service/src python3 -m pinn_service.cli \
  --materials /path/to/data_materials.csv \
  --temperature /path/to/data_temperature.csv \
  --displacement /path/to/data_displacement.csv \
  --stress1 /path/to/data_stress_1.csv \
  --stress2 /path/to/data_stress_2.csv \
  --stress3 /path/to/data_stress_3.csv \
  --output-dir /path/to/output \
  --build-training-matrix
```

## Strict 2D Retraining Workflow

The current practical 2D-only path is:

1. build strict 2D PINN artifacts under `pinn-service/artifacts/rod_experiments_2d`;
2. train `PINN` with `--physics-mode plane_strain_2d`;
3. batch-convert those 2D artifacts into per-rock `FNO`, `Transformer`, and `MGN` datasets;
4. switch `.env` to `baseline_2d` / `checkpoints_2d` checkpoints;
5. rerun direct model predictions and regenerate only 2D charts.

One-command 2D comparison refresh:

```bash
python scripts/run_2d_comparison_pipeline.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials_balanced.jsonl \
  --results-dir artifacts/data_experiments/results_2d_4materials_balanced \
  --figures-dir figures/results_2d_4materials_balanced \
  --tables-dir tables/results_2d_4materials_balanced
```

Batch FNO dataset build from strict 2D PINN artifacts:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/build_2d_fno_datasets.py \
  --input-root pinn-service/artifacts/rod_experiments_2d \
  --output-root fno-service/artifacts/datasets_2d \
  --grid-res 1 32 32 \
  --max-timesteps 64 \
  --validate
```

Batch Transformer dataset build:

```bash
PYTHONPATH=transformer-service/src python transformer-service/scripts/build_2d_transformer_datasets.py \
  --input-root pinn-service/artifacts/rod_experiments_2d \
  --output-root transformer-service/artifacts/datasets_2d
```

Batch MeshGraphNet dataset build:

```bash
python mgn-service/scripts/build_2d_mgn_datasets.py \
  --input-root pinn-service/artifacts/rod_experiments_2d \
  --registry-root mgn-service/datasets \
  --k-nearest 12 \
  --target-mode delta
```
