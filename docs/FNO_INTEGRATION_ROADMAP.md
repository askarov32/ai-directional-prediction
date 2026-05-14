# FNO Integration Roadmap

> Target repository: `https://github.com/askarov32/ai-directional-prediction`  
> Target file to add in the repo: `docs/FNO_INTEGRATION_ROADMAP.md`  
> Scope: planning only. Do not implement FNO code in this phase.

---

## 1. Current Repository Analysis

### 1.1 Current high-level architecture

The project is a thesis/demo MVP for AI directional prediction of thermoelastic wave propagation in geological media. The current stack is organized as a service-oriented application:

```text
frontend  ->  backend FastAPI orchestrator  ->  model services
                                      |-> meshgraphnet route
                                      |-> fno route
                                      |-> pinn route
                                      |-> transformer route / partially present in repo
```

Current top-level structure observed in the repository:

```text
.
├── .github/workflows/
├── analytics/
├── backend/
├── docs/
├── frontend/
├── granite-analytics/
├── mgn-service/
├── mock-services/
├── physics-theory/
├── pinn-service/
├── transformer-service/
├── .env.example
├── docker-compose.yml
├── pytest.ini
└── README.md
```

### 1.2 Current services

| Service | Current status | Notes |
|---|---|---|
| `frontend` | Real static frontend | Vanilla HTML/CSS/JS served by nginx. Calls backend through `/api/v1`. |
| `backend` | Real FastAPI orchestrator | Validates unified prediction requests, loads media catalog, routes to model clients, normalizes responses. |
| `mgn-service` | Dedicated MeshGraphNet service with fallback | Has a real-service shape and scripts. Can fallback if dataset/checkpoint is missing. |
| `mock-fno` | Mock service | Runs from `mock-services` with `SERVICE_KIND=fno`. This is the main target to replace. |
| `pinn-service` | Real checkpoint-based PINN service | Has COMSOL parsing, dataset building, training, inference, readiness diagnostics. |
| `transformer-service` | Present but not fully wired into Docker/backend | Has its own service folder, training artifacts, tests, and README; current scope says it is not wired into Docker Compose/backend router. |

### 1.3 Backend routing and model integration points

The backend already has the abstraction needed for FNO. The main files are:

```text
backend/app/domain/enums/model_type.py
backend/app/infrastructure/clients/base.py
backend/app/infrastructure/clients/fno_client.py
backend/app/infrastructure/clients/meshgraphnet_client.py
backend/app/infrastructure/clients/pinn_client.py
backend/app/domain/services/prediction_router.py
backend/app/domain/use_cases/predict_direction.py
backend/app/api/dependencies.py
backend/app/api/routes/predictions.py
backend/app/api/routes/models.py
backend/app/infrastructure/adapters/remote_response_schema.py
backend/app/infrastructure/adapters/response_normalizer.py
backend/app/schemas/prediction.py
backend/app/core/config.py
```

Current FNO flow:

1. Public request enters `POST /api/v1/predictions`.
2. `PredictionRequestSchema` validates payload.
3. `PredictDirectionUseCase` resolves `medium_id` and builds an `EnrichedPredictionRequest`.
4. `PredictionRouter` selects the client by `ModelType.FNO`.
5. `FNOClient.build_payload()` converts the unified backend entity to shared model-service payload and adds:

```json
{
  "representation": "grid",
  "routing_hint": "fno"
}
```

6. `BaseModelClient.predict()` sends `POST` to `MODEL_FNO_URL + MODEL_FNO_PREDICT_PATH`.
7. `RemoteModelResponse` validates the remote payload.
8. `ResponseNormalizer` produces the normalized frontend response.

This means the backend does not need a new model-selection system for FNO. The main integration change is to point the existing `FNOClient` to a real `fno-service` and optionally tighten the FNO-specific contract.

### 1.4 Current mock FNO

Current mock FNO is implemented through:

```text
mock-services/main.py
mock-services/common/predictor.py
```

`docker-compose.yml` runs it as:

```yaml
mock-fno:
  build:
    context: ./mock-services
  environment:
    SERVICE_KIND: fno
```

This service returns deterministic synthetic values and is not trainable. The roadmap should replace this route with a real `fno-service` while keeping a fallback path only for smoke/demo mode if needed.

### 1.5 Current PINN data pipeline

PINN is the most important existing data reference for FNO. Current files:

```text
pinn-service/src/pinn_service/comsol_parser.py
pinn-service/src/pinn_service/dataset_builder.py
pinn-service/src/pinn_service/training_data.py
pinn-service/src/pinn_service/training_config.py
pinn-service/src/pinn_service/train.py
pinn-service/src/pinn_service/trainer.py
pinn-service/src/pinn_service/model.py
pinn-service/src/pinn_service/losses.py
pinn-service/src/pinn_service/service_app.py
pinn-service/src/pinn_service/service_schemas.py
```

Current PINN input features:

```text
x, y, z, t, youngs_modulus, poissons_ratio, density,
thermal_expansion, thermal_conductivity, heat_capacity
```

Current PINN primary neural outputs:

```text
temperature_k, disp_x, disp_y, disp_z
```

Current PINN training matrix also contains additional targets:

```text
vel_x, vel_y, vel_z,
von_mises,
stress_x, stress_y, stress_z,
stress_xy, stress_yz, stress_xz,
strain_x, strain_y, strain_z,
strain_xy, strain_yz, strain_xz
```

PINN dataset artifacts:

```text
structured_dataset.npz
training_samples.npz
dataset_metadata.json
```

### 1.6 Existing universal data formatter relevant to FNO

There is already an important dataset bridge inside `mgn-service`:

```text
mgn-service/scripts/reformat_dataset.py
mgn-service/src/data/universal_formatter.py
```

It can produce:

```text
datasets/<dataset_id>/processed/canonical/
datasets/<dataset_id>/processed/graph/
datasets/<dataset_id>/processed/fno/
datasets/<dataset_id>/processed/pinn/
datasets/<dataset_id>/processed/transformer/
```

The FNO output produced by this formatter is especially useful:

```text
processed/fno/grid_dynamic.npy       # [T, C, Z, Y, X]
processed/fno/grid_static.npy        # [S, Z, Y, X]
processed/fno/grid_masks.npy         # [M, Z, Y, X]
processed/fno/grid_coords.npy        # [3, Z, Y, X]
processed/fno/source_node_index.npy
processed/fno/selected_time_indices.npy
processed/fno/field_names.json
processed/fno/static_feature_names.json
processed/fno/mask_names.json
processed/fno/metadata.json
```

This should be reused instead of creating a second incompatible FNO preprocessing pipeline. The cleanest MVP path is:

```text
COMSOL CSV / existing PINN-compatible data
  -> universal canonical format
  -> FNO regular grid adapter
  -> fno-service training and inference
```

### 1.7 Current tests and CI

Current test coverage exists for:

```text
backend/tests/
pinn-service/tests/
transformer-service/tests/
```

Current root `pytest.ini` includes:

```text
pythonpath = backend pinn-service/src
testpaths = backend/tests pinn-service/tests
```

Current GitHub Actions CI runs:

```text
backend tests
PINN tests
Docker Compose config validation
```

FNO tests should be added explicitly to `pytest.ini` and `.github/workflows/ci.yml`.

### 1.8 Current gaps

1. `fno-service/` does not exist as a real trainable service.
2. `docker-compose.yml` currently routes FNO to `mock-fno`, not a real FNO model.
3. Backend has `FNOClient`, but it expects only the normalized remote response shape; no strict FNO-specific service contract is documented yet.
4. The repository already has a universal FNO grid export path in `mgn-service`, but there is no FNO training/inference code consuming it.
5. Transformer service exists, but it is not fully wired into Docker/backend according to its README; do not couple FNO implementation to transformer wiring.
6. There is no documented FNO checkpoint format, metric format, readiness diagnostic format, or training command.

---

## 2. Target Architecture

### 2.1 Goal

Add a real `fno-service` that replaces `mock-fno` for the `fno` model route.

Target behavior:

```text
frontend
  -> backend /api/v1/predictions with model = "fno"
    -> backend FNOClient
      -> http://fno-service:9000/predict
        -> trained FNO checkpoint
        -> normalized backend-compatible JSON response
```

### 2.2 Target service map

```text
services:
  frontend
  backend
  mgn-service
  fno-service        # new real service
  pinn-service
  mock-fno           # remove from default compose or keep only as optional profile
```

Recommended compose strategy:

1. Replace default `mock-fno` with `fno-service`.
2. Keep `mock-fno` only behind a `demo-mocks` profile if the team still wants synthetic fallback.
3. Set `MODEL_FNO_URL=http://fno-service:9000` by default.

### 2.3 FNO role

`fno-service` should own:

- FNO model definition.
- Spectral convolution layers.
- FNO-compatible dataset loading.
- Training loop.
- Evaluation loop.
- Checkpoint saving/loading.
- Inference API.
- Readiness diagnostics.
- FNO-specific tests.
- FNO documentation.

It should not own:

- Public product API validation.
- Geological medium catalog.
- Frontend normalization.
- Backend routing logic already present in `backend`.

### 2.4 Data path

Recommended data path for MVP:

```text
Raw COMSOL exports / existing dataset
  -> existing PINN builder or universal formatter
  -> regular FNO grid tensors
  -> fno-service dataset loader
  -> FNO training
  -> checkpoint
  -> inference endpoint
```

Preferred source for FNO tensors:

```text
mgn-service/scripts/reformat_dataset.py --formats fno
```

Alternative fallback if universal formatter is not stable enough:

```text
pinn-service/artifacts/<dataset>/structured_dataset.npz
  -> fno-service/src/fno_service/data/pinn_to_grid.py
  -> fno-service/artifacts/datasets/<dataset>/fno_grid.npz
```

For MVP, prefer the universal formatter path because it already has explicit FNO grid output.

### 2.5 Backend compatibility target

The real FNO service should return a shape already accepted by `RemoteModelResponse`:

```json
{
  "prediction": {
    "direction_vector": [0.0, 1.0, 0.0],
    "azimuth_deg": 90.0,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "fno_coupled_field",
    "travel_time_ms": 12.3
  },
  "field_summary": {
    "max_displacement": 0.001,
    "max_temperature_perturbation": 1.2
  },
  "model_version": "fno-baseline@best_model.pth"
}
```

Extra fields are allowed but must not break the backend normalizer:

```json
{
  "diagnostics": {},
  "grid_summary": {},
  "checkpoint": {},
  "postprocessed_prediction": {}
}
```

---

## 3. New `fno-service` Structure

Recommended structure:

```text
fno-service/
  README.md
  Dockerfile
  requirements.txt
  configs/
    default.yaml
    train_fno.yaml
    inference.yaml
  src/
    fno_service/
      __init__.py
      api/
        __init__.py
        main.py
        schemas.py
        routes.py
      data/
        __init__.py
        dataset.py
        preprocessing.py
        dataloaders.py
        pinn_to_grid.py
      models/
        __init__.py
        fno.py
        layers.py
      training/
        __init__.py
        trainer.py
        losses.py
        metrics.py
        checkpoints.py
      inference/
        __init__.py
        predictor.py
        postprocessing.py
      utils/
        __init__.py
        config.py
        logging.py
        seed.py
  scripts/
    train_fno.py
    evaluate_fno.py
    run_inference.py
    prepare_fno_dataset.py
  tests/
    test_dataset.py
    test_preprocessing.py
    test_model.py
    test_training.py
    test_api.py
    test_inference.py
  artifacts/
    .gitkeep
```

### Why this structure

This mirrors the current `pinn-service` and `transformer-service` style while keeping FNO-specific concerns isolated. The package name should be `fno_service` under `src/` to avoid top-level import collisions and to work with `PYTHONPATH=fno-service/src`.

### Main responsibilities by module

| Module | Responsibility |
|---|---|
| `api/main.py` | FastAPI app construction. |
| `api/routes.py` | `/health`, `/ready`, `/predict`. |
| `api/schemas.py` | Request/response Pydantic schemas. |
| `data/dataset.py` | Load FNO grid tensors and expose Torch dataset. |
| `data/preprocessing.py` | Build input channels, normalize, split time windows. |
| `data/dataloaders.py` | Create train/val/test DataLoaders. |
| `data/pinn_to_grid.py` | Fallback conversion from PINN `structured_dataset.npz` to FNO grid. |
| `models/layers.py` | `SpectralConv2d`, `SpectralConv3d` or unified spectral layer. |
| `models/fno.py` | FNO2D/FNO3D model. |
| `training/trainer.py` | Training loop, validation loop, checkpoint calls. |
| `training/losses.py` | Supervised and optional physics-inspired losses. |
| `training/metrics.py` | MAE/RMSE/relative L2/angular/travel-time metrics. |
| `training/checkpoints.py` | Save/load model, scalers, config, metrics. |
| `inference/predictor.py` | Load checkpoint and run model inference. |
| `inference/postprocessing.py` | Convert field prediction into direction summary. |
| `utils/config.py` | YAML config loader with dataclasses/Pydantic validation. |
| `utils/seed.py` | Deterministic seed setup. |

---

## 4. File-by-file Change Plan

| File | Action | What changes | Why |
|---|---|---|---|
| `docs/FNO_INTEGRATION_ROADMAP.md` | create | Add this roadmap. | Required planning artifact before implementation. |
| `fno-service/README.md` | create | Document service purpose, data format, training, inference, Docker usage, troubleshooting. | Make FNO service self-contained. |
| `fno-service/Dockerfile` | create | Python 3.11 image, install requirements, non-root user, expose `9000`, default uvicorn command. | Run FNO service in Docker Compose. |
| `fno-service/requirements.txt` | create | Add `numpy`, `torch`, `fastapi`, `pydantic`, `pytest`, `uvicorn[standard]`, `PyYAML`; optionally `scipy`/`scikit-learn` only if needed. | Provide reproducible FNO runtime/training deps. |
| `fno-service/configs/default.yaml` | create | Shared model/data/training defaults. | Central config baseline. |
| `fno-service/configs/train_fno.yaml` | create | Dataset path, grid dimensions, channels, epochs, batch size, learning rate, checkpoint output. | Training command should be config-driven. |
| `fno-service/configs/inference.yaml` | create | Checkpoint path, device, readiness settings, postprocessing parameters. | Separate training and inference settings. |
| `fno-service/src/fno_service/__init__.py` | create | Package marker and version constant. | Clean imports. |
| `fno-service/src/fno_service/api/main.py` | create | Instantiate FastAPI and include routes. | Service entry point for uvicorn. |
| `fno-service/src/fno_service/api/routes.py` | create | Implement `GET /health`, `GET /ready`, `POST /predict`. | Backend expects health/readiness/predict endpoints. |
| `fno-service/src/fno_service/api/schemas.py` | create | Define `PredictionPayload`, `PredictionResponse`, `ReadinessResponse`, field summaries. | Enforce API contract. |
| `fno-service/src/fno_service/data/dataset.py` | create | Load `grid_dynamic.npy`, `grid_static.npy`, `grid_masks.npy`, metadata; return time-window samples. | FNO needs grid tensor batches. |
| `fno-service/src/fno_service/data/preprocessing.py` | create | Build input channels: current field state + static material grid + masks + coordinates/time. | Convert stored tensors to model-ready input. |
| `fno-service/src/fno_service/data/dataloaders.py` | create | Create train/val/test dataloaders from time-index splits. | Keep trainer simple and testable. |
| `fno-service/src/fno_service/data/pinn_to_grid.py` | create | Optional fallback converter from PINN `structured_dataset.npz` into regular grid using nearest-neighbor or simple binning. | Satisfy requirement to use same/compatible data as PINN if universal formatter is unavailable. |
| `fno-service/src/fno_service/models/layers.py` | create | Implement spectral convolution layer using `torch.fft.rfft2`/`irfft2` and optionally 3D `rfftn`/`irfftn`. | Core FNO operation. |
| `fno-service/src/fno_service/models/fno.py` | create | Implement `FNO2d` and optionally `FNO3d`; configurable modes, width, depth, in/out channels. | Main model. |
| `fno-service/src/fno_service/training/trainer.py` | create | Training loop, validation, scheduler, early stopping, metric logging. | Real training pipeline. |
| `fno-service/src/fno_service/training/losses.py` | create | MSE field loss, masked loss, relative L2; optional smoothness/temporal consistency loss. | FNO objective. |
| `fno-service/src/fno_service/training/metrics.py` | create | RMSE, MAE, relative L2, per-channel metrics, final direction angular error helper if labels exist. | Evaluation and reporting. |
| `fno-service/src/fno_service/training/checkpoints.py` | create | Save/load `model.pth`, `best_model.pth`, `metrics.json`, `training_config.json`, `scalers.json`, `metadata.json`. | Reproducible training and inference. |
| `fno-service/src/fno_service/inference/predictor.py` | create | Load checkpoint, construct model, run inference, apply postprocessing. | Runtime inference logic. |
| `fno-service/src/fno_service/inference/postprocessing.py` | create | Convert predicted fields into backend response: direction vector, azimuth/elevation, travel time, max displacement/temperature. | Backend needs directional summary, not only raw grid fields. |
| `fno-service/src/fno_service/utils/config.py` | create | Load and validate YAML configs. | Avoid untyped config dicts. |
| `fno-service/src/fno_service/utils/logging.py` | create | Consistent logger setup. | Diagnostics. |
| `fno-service/src/fno_service/utils/seed.py` | create | Set Python/NumPy/Torch seeds. | Reproducibility. |
| `fno-service/scripts/prepare_fno_dataset.py` | create | CLI to build FNO grid from universal formatter output or PINN structured dataset. | Reproducible dataset preparation. |
| `fno-service/scripts/train_fno.py` | create | CLI entry for training from YAML. | Required training command. |
| `fno-service/scripts/evaluate_fno.py` | create | Evaluate checkpoint on validation/test split. | Model validation. |
| `fno-service/scripts/run_inference.py` | create | Local one-off inference from JSON payload/checkpoint. | Debug without backend. |
| `fno-service/tests/test_dataset.py` | create | Validate tensor loading, shapes, splits, finite values. | Unit coverage for data pipeline. |
| `fno-service/tests/test_preprocessing.py` | create | Validate channel construction, masking, normalization. | Prevent silent shape bugs. |
| `fno-service/tests/test_model.py` | create | Forward pass for 2D/3D small tensors; output shape and finite values. | Core model safety. |
| `fno-service/tests/test_training.py` | create | One or two batches over synthetic tiny dataset; checkpoint created. | Smoke-test training. |
| `fno-service/tests/test_api.py` | create | Test `/health`, `/ready`, `/predict` with missing and dummy checkpoint states. | API contract coverage. |
| `fno-service/tests/test_inference.py` | create | Load tiny checkpoint and produce backend-compatible response. | Runtime safety. |
| `docker-compose.yml` | update | Add `fno-service`; change backend `MODEL_FNO_URL` default to `http://fno-service:9000`; make backend depend on `fno-service`; move `mock-fno` to optional profile or remove it from default. | Replace mock FNO with real FNO service. |
| `.env.example` | update | Add `FNO_SERVICE_PORT`, `FNO_CHECKPOINT_PATH`, `FNO_DEVICE`, `FNO_CONFIG_PATH`, `FNO_DATASET_ID`, `FNO_ALLOW_FALLBACK`, `FNO_LOG_LEVEL`; change `MODEL_FNO_URL`. | Expose FNO runtime config. |
| `backend/app/core/config.py` | update | Add optional FNO-specific timeout if needed: `model_fno_timeout_seconds`; keep generic timeout if not needed. Validate `model_fno_url`. | Support heavier FNO inference without touching other models. |
| `backend/app/api/dependencies.py` | update | Pass FNO-specific timeout only if added. No structural change needed otherwise. | Wire config to `FNOClient`. |
| `backend/app/infrastructure/clients/fno_client.py` | update | Keep `representation=grid`; optionally add `requested_outputs`, `grid_policy`, `routing_hint=fno`. | Make backend payload explicit for real FNO. |
| `backend/app/infrastructure/adapters/remote_response_schema.py` | update | Usually no change. Only update if FNO response needs extra strict fields; keep `extra="ignore"`. | Preserve backend compatibility. |
| `backend/app/domain/enums/model_type.py` | update/no-op | No change required if only `meshgraphnet`, `fno`, `pinn` are supported. Add `TRANSFORMER` only as separate task if transformer route is intentionally wired. | Avoid unrelated routing changes. |
| `backend/tests/test_predict_direction_use_case.py` | update | Add or adjust FNO route assertions to target real-service response shape. | Ensure backend uses FNO client correctly. |
| `backend/tests/test_response_normalizer.py` | update | Add FNO payload case with `prediction` + `field_summary`. | Protect response contract. |
| `backend/tests/test_model_client_errors.py` | update | Add FNO timeout/unavailable/malformed response cases if not already generic. | Validate error handling. |
| `backend/tests/test_api_smoke.py` | update | Include `model=fno` prediction smoke test with mocked service/client. | API coverage. |
| `pytest.ini` | update | Add `fno-service/src` to `pythonpath` and `fno-service/tests` to `testpaths`. | Run FNO tests locally. |
| `.github/workflows/ci.yml` | update | Add `fno-tests` job installing `fno-service/requirements.txt` and running `pytest fno-service/tests`; optionally add docker compose config check with new service. | CI coverage. |
| `README.md` | update | Replace “mock FNO” status with real `fno-service` status and document commands. | Keep top-level docs accurate. |
| `docs/ai_thermoelastic_architecture.md` | update | Add real FNO service to service map, Docker endpoints, and model routing notes. | Architecture docs. |
| `docs/model_card.md` | update | Update FNO section from mock to baseline trained FNO, with limitations. | Scientific honesty. |
| `docs/demo_limitations.md` | update | Remove or soften “FNO is mocked” after real service is implemented; add validation caveats. | Demo limitations must match implementation. |
| `docs/prediction_contract_and_training_formulas.md` | update | Add FNO service payload/response contract and FNO training objective summary. | API and training docs. |
| `docs/FNO_SERVICE.md` | create | Service overview, architecture, endpoints, env vars, readiness. | Dedicated FNO service docs. |
| `docs/FNO_TRAINING.md` | create | Dataset prep, training commands, config keys, artifacts, metrics. | Training guide. |
| `docs/FNO_API_CONTRACT.md` | create | Backend-to-FNO request/response JSON contract with examples. | Contract docs. |
| `docs/FNO_DATASET_FORMAT.md` | create | Document grid tensors, channel names, metadata, normalization. | Dataset reproducibility. |
| `docs/MODEL_SERVICES_OVERVIEW.md` | create/update | Summarize PINN/MGN/FNO/Transformer routes and status. | Prevent architecture drift. |
| `mock-services/main.py` | update/no-op | Keep for optional mock profile only. Do not delete unless explicitly agreed. | Preserve demo fallback. |
| `mock-services/common/predictor.py` | update/no-op | Keep FNO mock generator only if `mock-fno` remains as optional profile. | Backward compatibility. |

---

## 5. Training Pipeline

### 5.1 Primary dataset strategy

Use the same COMSOL-origin data family as PINN, but train FNO on grid tensors.

Preferred path:

```bash
python mgn-service/scripts/reformat_dataset.py \
  --config mgn-service/configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats canonical fno \
  --grid_res 32 32 32 \
  --fno_max_timesteps 128 \
  --fno_normalization normalized
```

Expected input for `fno-service`:

```text
mgn-service/datasets/<dataset_id>/processed/fno/
  grid_dynamic.npy
  grid_static.npy
  grid_masks.npy
  grid_coords.npy
  source_node_index.npy
  selected_time_indices.npy
  field_names.json
  static_feature_names.json
  mask_names.json
  metadata.json
```

Alternative path for strict PINN compatibility:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/prepare_fno_dataset.py \
  --pinn-structured pinn-service/artifacts/demo/structured_dataset.npz \
  --pinn-metadata pinn-service/artifacts/demo/dataset_metadata.json \
  --output-dir fno-service/artifacts/datasets/demo_fno \
  --grid-res 32 32 32
```

### 5.2 Dataset class

Create:

```text
fno-service/src/fno_service/data/dataset.py
```

Recommended dataset classes:

```text
FNOGridDataset
FNOTimeStepDataset
FNOAutoregressiveDataset
```

MVP choice:

```text
FNOTimeStepDataset
```

Each sample should represent:

```text
input  = [current_dynamic_fields, static_fields, masks, coords, time_channel]
target = next_dynamic_fields or selected target fields
```

Recommended shapes for 3D:

```text
grid_dynamic: [T, C_dyn, Z, Y, X]
grid_static:  [C_static, Z, Y, X]
grid_masks:   [C_mask, Z, Y, X]
grid_coords:  [3, Z, Y, X]
input:        [C_in, Z, Y, X]
target:       [C_out, Z, Y, X]
```

Recommended shapes for 2D fallback:

```text
grid_dynamic: [T, C_dyn, Y, X]
grid_static:  [C_static, Y, X]
grid_masks:   [C_mask, Y, X]
grid_coords:  [2, Y, X]
input:        [C_in, Y, X]
target:       [C_out, Y, X]
```

### 5.3 Input channels

Recommended MVP input channels:

```text
current dynamic fields:
  temperature_k
  disp_x
  disp_y
  disp_z

static material fields:
  youngs_modulus / E
  poissons_ratio / nu
  density / rho
  thermal_expansion / alpha
  thermal_conductivity / k
  heat_capacity / Cp

geometry and masks:
  x_grid
  y_grid
  z_grid
  source_mask
  boundary masks
  valid_grid_mask

time:
  normalized time t
```

If the universal formatter names differ from PINN names, `preprocessing.py` must map aliases explicitly.

Example aliases:

| Canonical/PINN concept | Possible field names |
|---|---|
| Temperature | `temperature_k`, `T`, `temperature` |
| Displacement x | `disp_x`, `u` |
| Displacement y | `disp_y`, `v` |
| Displacement z | `disp_z`, `w` |
| Young modulus | `youngs_modulus`, `E`, `solid.E` |
| Poisson ratio | `poissons_ratio`, `nu`, `solid.nu` |
| Density | `density`, `rho`, `solid.rho`, `thermal_density` |
| Thermal expansion | `thermal_expansion`, `alpha` |
| Thermal conductivity | `thermal_conductivity`, `k` |
| Heat capacity | `heat_capacity`, `Cp` |

### 5.4 Target channels

MVP target channels:

```text
temperature_k
disp_x
disp_y
disp_z
```

Optional later target channels:

```text
vel_x, vel_y, vel_z
von_mises
stress components
strain components
```

Do not train all 20 PINN targets in the first FNO MVP unless memory and data quality are verified. Start with the same primary outputs used by the PINN network: temperature and displacement.

### 5.5 Model objective

Recommended MVP loss:

```text
L_total = L_field + lambda_mask * L_masked + lambda_smooth * L_smooth
```

Where:

```text
L_field  = MSE(predicted fields, target fields)
L_masked = MSE over valid grid/source/boundary mask areas, optional
L_smooth = optional spatial smoothness regularization, low weight
```

Default MVP weights:

```yaml
loss:
  field_weight: 1.0
  masked_weight: 0.0
  smoothness_weight: 0.0
```

Start simple. Add physics-inspired terms only after baseline field learning works.

### 5.6 Metrics

Training metrics:

```text
train_loss
val_loss
per_channel_mae
per_channel_rmse
per_channel_relative_l2
```

Prediction/postprocessing metrics:

```text
max_displacement
max_temperature_perturbation
travel_time_ms
```

If reference direction labels exist later:

```text
angular_error_deg
azimuth_error_deg
elevation_error_deg
```

### 5.7 Checkpoint artifacts

Training output directory:

```text
fno-service/artifacts/checkpoints/baseline/
  best_model.pth
  model.pth
  metrics.json
  metrics.csv
  training_config.json
  dataset_metadata.json
  channel_metadata.json
  scalers.json
```

Checkpoint payload should include:

```python
{
  "model_state_dict": ...,
  "model_config": ...,
  "training_config": ...,
  "input_channel_names": ...,
  "output_channel_names": ...,
  "normalization": ...,
  "dataset_metadata": ...,
  "best_val_loss": ...,
  "epoch": ...,
}
```

### 5.8 CLI commands

Dataset preparation:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/prepare_fno_dataset.py \
  --source mgn-service/datasets/sandstone_comsol_real/processed/fno \
  --output-dir fno-service/artifacts/datasets/sandstone_fno \
  --validate
```

Training:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/train_fno.py \
  --config fno-service/configs/train_fno.yaml
```

Smoke training:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/train_fno.py \
  --config fno-service/configs/train_fno.yaml \
  --epochs 1 \
  --sample-limit 4 \
  --batch-size 1 \
  --device cpu
```

Evaluation:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/evaluate_fno.py \
  --checkpoint fno-service/artifacts/checkpoints/baseline/best_model.pth \
  --dataset fno-service/artifacts/datasets/sandstone_fno
```

Local inference:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/run_inference.py \
  --checkpoint fno-service/artifacts/checkpoints/baseline/best_model.pth \
  --payload examples/fno_prediction_request.json
```

Docker training:

```bash
docker compose up -d fno-service
docker compose exec fno-service python scripts/train_fno.py --config configs/train_fno.yaml
```

---

## 6. API Integration Plan

### 6.1 Backend-to-FNO request contract

The backend already sends an enriched payload with:

```json
{
  "medium": {
    "id": "sandstone_medium",
    "name": "Sandstone",
    "category": "sedimentary",
    "properties": {},
    "ranges": {},
    "metadata": {}
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
  "representation": "grid",
  "routing_hint": "fno"
}
```

Recommended small addition in `FNOClient.build_payload()`:

```json
{
  "requested_outputs": ["direction", "field_summary"],
  "grid_policy": "service_default"
}
```

Do not send full grid tensors through the backend request. The service should load its dataset/checkpoint artifacts locally and use the request only as conditioning/scenario/probe input.

### 6.2 FNO response contract

Return backend-compatible shape:

```json
{
  "prediction": {
    "direction_vector": [0.821, 0.571, 0.0],
    "azimuth_deg": 34.8,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "fno_coupled_field",
    "travel_time_ms": 12.4
  },
  "field_summary": {
    "max_displacement": 0.001327,
    "max_temperature_perturbation": 1.742
  },
  "model_version": "fno-baseline@best_model.pth",
  "diagnostics": {
    "checkpoint_loaded": true,
    "device": "cpu",
    "input_channels": [],
    "output_channels": []
  }
}
```

### 6.3 FNO endpoints

Implement:

```text
GET /health
GET /ready
POST /predict
```

`GET /health` should not require checkpoint:

```json
{
  "status": "ok",
  "service": "fno"
}
```

`GET /ready` should verify:

- checkpoint path resolved;
- checkpoint file exists;
- model config loaded;
- model weights loaded;
- input/output channel metadata loaded;
- smoke inference passes;
- output is finite and shape-valid.

If missing checkpoint:

- `GET /health` returns `200` with `ready=false` details or plain status ok.
- `GET /ready` returns `503` unless `FNO_ALLOW_FALLBACK=true`.
- `POST /predict` returns `503 CHECKPOINT_NOT_READY` unless fallback is explicitly enabled.

### 6.4 Error handling

Service-level errors:

| Condition | HTTP status | Error code |
|---|---:|---|
| Missing checkpoint | 503 | `CHECKPOINT_NOT_READY` |
| Invalid request shape | 422 | FastAPI/Pydantic validation |
| Unsupported domain/grid | 400 | `UNSUPPORTED_DOMAIN` |
| Model output contains NaN/Inf | 500 | `NON_FINITE_MODEL_OUTPUT` |
| Inference timeout | 504 | `INFERENCE_TIMEOUT` |
| Internal model load error | 500 | `MODEL_LOAD_FAILED` |

Backend already wraps common HTTP/timeout/malformed-response failures through `BaseModelClient`. Keep the FNO response inside the existing `RemoteModelResponse` contract.

### 6.5 Backend files to change

Minimal required backend changes:

```text
backend/app/infrastructure/clients/fno_client.py
backend/tests/test_predict_direction_use_case.py
backend/tests/test_response_normalizer.py
backend/tests/test_model_client_errors.py
backend/tests/test_api_smoke.py
```

Optional backend changes:

```text
backend/app/core/config.py
backend/app/api/dependencies.py
```

Do not rewrite `PredictionRouter` unless the model enum/router is intentionally expanded.

---

## 7. Docker/DevOps Changes

### 7.1 `docker-compose.yml`

Target service:

```yaml
fno-service:
  build:
    context: ./fno-service
  container_name: fno-service
  environment:
    FNO_CHECKPOINT_PATH: ${FNO_CHECKPOINT_PATH:-/app/artifacts/checkpoints/baseline}
    FNO_CONFIG_PATH: ${FNO_CONFIG_PATH:-/app/configs/inference.yaml}
    FNO_DATASET_PATH: ${FNO_DATASET_PATH:-/app/artifacts/datasets/sandstone_fno}
    FNO_DEVICE: ${FNO_DEVICE:-cpu}
    FNO_LOG_LEVEL: ${FNO_LOG_LEVEL:-INFO}
    FNO_SERVICE_PORT: 9000
    FNO_ALLOW_FALLBACK: ${FNO_ALLOW_FALLBACK:-false}
  volumes:
    - ./fno-service/artifacts:/app/artifacts
    - ./mgn-service/datasets:/app/datasets:ro
  ports:
    - "${FNO_SERVICE_PORT:-9002}:9000"
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/ready')"]
    interval: 15s
    timeout: 5s
    retries: 5
    start_period: 10s
```

Update backend environment:

```yaml
MODEL_FNO_URL: ${MODEL_FNO_URL:-http://fno-service:9000}
MODEL_FNO_PREDICT_PATH: ${MODEL_FNO_PREDICT_PATH:-/predict}
```

Update backend `depends_on`:

```yaml
depends_on:
  fno-service:
    condition: service_healthy
```

Optional mock profile:

```yaml
mock-fno:
  profiles: ["demo-mocks"]
```

### 7.2 `.env.example`

Add/update:

```text
FNO_SERVICE_PORT=9002
FNO_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline
FNO_CONFIG_PATH=/app/configs/inference.yaml
FNO_DATASET_PATH=/app/artifacts/datasets/sandstone_fno
FNO_DEVICE=cpu
FNO_LOG_LEVEL=INFO
FNO_ALLOW_FALLBACK=false
MODEL_FNO_URL=http://fno-service:9000
MODEL_FNO_PREDICT_PATH=/predict
```

### 7.3 Dockerfile

Recommended `fno-service/Dockerfile` shape:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app appuser

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/artifacts && chown -R appuser:app /app
USER appuser

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "fno_service.api.main:app", "--host", "0.0.0.0", "--port", "9000"]
```

### 7.4 Local validation commands

```bash
docker compose config --quiet
docker compose build fno-service
docker compose up fno-service
curl -s http://localhost:9002/health
curl -s http://localhost:9002/ready
```

Full stack:

```bash
docker compose up --build
curl -s http://localhost:8000/api/v1/ready
curl -s http://localhost:8080/api/v1/models
```

FNO prediction through backend:

```bash
curl -s -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data @examples/fno_prediction_request.json
```

---

## 8. Testing Plan

### 8.1 Unit tests for `fno-service`

Create:

```text
fno-service/tests/test_dataset.py
fno-service/tests/test_preprocessing.py
fno-service/tests/test_model.py
fno-service/tests/test_training.py
fno-service/tests/test_inference.py
```

Coverage:

- Load synthetic `grid_dynamic.npy`, `grid_static.npy`, `grid_masks.npy`, metadata.
- Validate tensor dimensions.
- Validate 2D and 3D forward pass.
- Validate spectral layer output shape.
- Validate masked loss ignores invalid grid cells.
- Validate metrics are finite.
- Validate checkpoint save/load roundtrip.
- Validate one-batch training creates `model.pth` and `metrics.json`.

### 8.2 API tests for `fno-service`

Create:

```text
fno-service/tests/test_api.py
```

Coverage:

- `GET /health` returns 200.
- `GET /ready` returns ready payload if checkpoint exists.
- `GET /ready` returns 503 if checkpoint missing and fallback disabled.
- `POST /predict` validates request schema.
- `POST /predict` returns backend-compatible JSON.
- Invalid domain/request gives 422 or 400.
- Missing checkpoint gives 503.

### 8.3 Backend integration tests

Update:

```text
backend/tests/test_predict_direction_use_case.py
backend/tests/test_response_normalizer.py
backend/tests/test_model_client_errors.py
backend/tests/test_api_smoke.py
```

Coverage:

- `model=fno` routes to `FNOClient`.
- `FNOClient.build_payload()` includes `representation=grid` and `routing_hint=fno`.
- Backend normalizer accepts FNO response with nested `prediction` and `field_summary`.
- Backend handles FNO timeout/unavailable/malformed response.
- Public API response schema remains stable.

### 8.4 CI changes

Update `pytest.ini`:

```ini
[pytest]
pythonpath = backend pinn-service/src fno-service/src
testpaths = backend/tests pinn-service/tests fno-service/tests
addopts = -q
```

Update `.github/workflows/ci.yml`:

```yaml
fno-tests:
  name: FNO tests
  runs-on: ubuntu-latest
  env:
    PYTHONDONTWRITEBYTECODE: "1"
    PYTHONPATH: fno-service/src
  steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
        cache-dependency-path: fno-service/requirements.txt
    - name: Install FNO dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install -r fno-service/requirements.txt
    - name: Run FNO tests
      run: python -m pytest fno-service/tests
```

Do not add full model training to CI. Keep CI training to a tiny synthetic smoke test only.

---

## 9. Documentation Plan

### 9.1 Create new docs

```text
docs/FNO_INTEGRATION_ROADMAP.md
docs/FNO_SERVICE.md
docs/FNO_TRAINING.md
docs/FNO_API_CONTRACT.md
docs/FNO_DATASET_FORMAT.md
docs/MODEL_SERVICES_OVERVIEW.md
```

### 9.2 Update existing docs

```text
README.md
docs/ai_thermoelastic_architecture.md
docs/model_card.md
docs/demo_limitations.md
docs/prediction_contract_and_training_formulas.md
```

### 9.3 Required documentation content

`docs/FNO_SERVICE.md`:

- Service purpose.
- Runtime env vars.
- API endpoints.
- Readiness behavior.
- Checkpoint loading.
- Error responses.

`docs/FNO_TRAINING.md`:

- Dataset preparation.
- FNO grid format.
- Training config keys.
- Training commands.
- Smoke training.
- Checkpoint artifacts.
- Evaluation commands.

`docs/FNO_API_CONTRACT.md`:

- Backend-to-FNO request JSON.
- FNO-to-backend response JSON.
- Error contract.
- Compatibility notes with `RemoteModelResponse`.

`docs/FNO_DATASET_FORMAT.md`:

- `grid_dynamic.npy` shape.
- `grid_static.npy` shape.
- `grid_masks.npy` shape.
- `grid_coords.npy` shape.
- Metadata files.
- Channel names.
- Normalization rules.

`docs/MODEL_SERVICES_OVERVIEW.md`:

- Current model route status: PINN, MGN, FNO, Transformer.
- Which services are real, fallback, or mock.
- Docker ports.
- Health/readiness routes.

---

## 10. Implementation Phases

### Phase 1 — Repository audit and architecture alignment

Tasks:

- Confirm current `docker-compose.yml` service names and ports.
- Confirm current backend `ModelType` values.
- Confirm FNO route currently goes through `FNOClient` and `MODEL_FNO_URL`.
- Confirm `mock-fno` is the current default FNO target.
- Confirm PINN dataset artifacts and universal FNO grid artifacts.
- Decide whether FNO service reads `mgn-service/datasets/.../processed/fno` directly or copies into `fno-service/artifacts/datasets`.

Deliverable:

```text
docs/FNO_INTEGRATION_ROADMAP.md
```

### Phase 2 — Create `fno-service` skeleton

Tasks:

- Add folder structure.
- Add Dockerfile.
- Add requirements.
- Add FastAPI app.
- Add `/health`, `/ready`, `/predict` placeholders.
- Add config loader.
- Add basic tests for service startup.

Validation:

```bash
PYTHONPATH=fno-service/src python -m pytest fno-service/tests/test_api.py
```

### Phase 3 — Dataset compatibility with PINN/universal data

Tasks:

- Implement `FNOGridDataset`.
- Load universal FNO artifacts.
- Implement metadata/channel validation.
- Add optional PINN `structured_dataset.npz` to grid converter.
- Add synthetic tiny dataset fixture for tests.

Validation:

```bash
PYTHONPATH=fno-service/src python -m pytest fno-service/tests/test_dataset.py fno-service/tests/test_preprocessing.py
```

### Phase 4 — FNO model implementation

Tasks:

- Implement `SpectralConv2d`.
- Implement `FNO2d`.
- Add `SpectralConv3d`/`FNO3d` only if current data is confirmed 3D and memory allows.
- Make modes/width/depth configurable.
- Add CPU/CUDA handling.

Validation:

```bash
PYTHONPATH=fno-service/src python -m pytest fno-service/tests/test_model.py
```

### Phase 5 — Training pipeline

Tasks:

- Implement losses.
- Implement metrics.
- Implement trainer.
- Implement checkpoint save/load.
- Implement `scripts/train_fno.py`.
- Implement smoke training mode.

Validation:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/train_fno.py \
  --config fno-service/configs/train_fno.yaml \
  --epochs 1 \
  --sample-limit 4 \
  --device cpu
```

### Phase 6 — Inference API

Tasks:

- Implement checkpoint loading in `Predictor`.
- Implement smoke inference in readiness.
- Implement postprocessing to backend-compatible direction response.
- Implement missing-checkpoint errors.
- Add `scripts/run_inference.py`.

Validation:

```bash
curl -s http://localhost:9002/ready
curl -s -X POST http://localhost:9002/predict -H "Content-Type: application/json" --data @examples/fno_prediction_request.json
```

### Phase 7 — Backend integration

Tasks:

- Change compose backend `MODEL_FNO_URL` to `http://fno-service:9000`.
- Add backend `depends_on` for `fno-service`.
- Keep `FNOClient` minimal and compatible.
- Add backend tests for real FNO response shape.
- Keep PINN/MGN behavior unchanged.

Validation:

```bash
python -m pytest backend/tests
curl -s -X POST http://localhost:8000/api/v1/predictions -H "Content-Type: application/json" --data @examples/fno_prediction_request.json
```

### Phase 8 — Tests

Tasks:

- Add FNO unit tests.
- Add FNO API tests.
- Add backend integration tests.
- Add smoke-training test.
- Update `pytest.ini`.
- Update GitHub Actions.

Validation:

```bash
python -m pytest
```

### Phase 9 — Documentation

Tasks:

- Add FNO docs.
- Update top-level README.
- Update model card.
- Update demo limitations.
- Update architecture docs.
- Add FNO contract docs.

Validation:

- Docs commands match actual commands.
- Docs no longer say FNO is only mocked after real service is implemented.

### Phase 10 — Final validation

Tasks:

- Run backend tests.
- Run PINN tests.
- Run FNO tests.
- Run Docker Compose config check.
- Run FNO training smoke test.
- Run FNO `/ready`.
- Run backend → FNO prediction request.
- Verify frontend can select `fno` and receive normalized output.

Commands:

```bash
python -m pytest
docker compose config --quiet
docker compose up --build
curl -s http://localhost:8000/api/v1/ready
curl -s http://localhost:8080/api/v1/models
curl -s -X POST http://localhost:8080/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data @examples/fno_prediction_request.json
```

---

## 11. Risks and Open Questions

### 11.1 Data/grid mismatch

Risk:

FNO requires regular grid tensors, while COMSOL/PINN data may be node/point based and not naturally regular.

Mitigation:

- Use existing `mgn-service` universal formatter FNO export.
- Start with nearest-neighbor grid interpolation already present in the formatter.
- Keep `grid_valid_mask` so loss ignores unreliable cells.

### 11.2 Memory usage

Risk:

Full 3D FNO grids can be very large.

Mitigation:

- Start with `32x32x32` or smaller.
- Limit timesteps with `--fno_max_timesteps 128`.
- Add 2D mode for quick smoke tests.
- Use tiny synthetic tensors in CI.

### 11.3 Directional postprocessing may be weak

Risk:

FNO predicts fields, but backend needs direction/travel-time summary.

Mitigation:

- Implement explicit `postprocessing.py`.
- For MVP, compute direction from source/probe geometry and refine with predicted field gradients/temperature/displacement maxima.
- Document that directional inference is MVP postprocessing, not final scientific validation.

### 11.4 Scientific validation not complete

Risk:

A trained FNO can run but not be scientifically validated.

Mitigation:

- Document limitations.
- Report field RMSE/MAE/relative L2.
- Add direction angular error only when reference direction labels exist.
- Avoid claims that FNO is production-grade.

### 11.5 Existing transformer route ambiguity

Risk:

README mentions `transformer`, but backend enum currently may not include it in the active route list, and transformer README says it is not wired into Docker/backend.

Mitigation:

- Do not expand transformer routing as part of FNO work.
- Preserve transformer files.
- Track transformer wiring as a separate task.

### 11.6 Mock service removal may break demo workflows

Risk:

Removing `mock-fno` could break quick demos if FNO checkpoint is missing.

Mitigation:

- Keep `mock-fno` behind a Compose profile.
- Add `FNO_ALLOW_FALLBACK=false` by default for correctness.
- Allow explicit fallback for demos only.

### 11.7 Checkpoint availability

Risk:

A real checkpoint may be too large for Git or unavailable locally.

Mitigation:

- Do not commit large checkpoints.
- Commit `.gitkeep` only.
- Document how to train baseline checkpoint.
- Add tiny test checkpoint fixture only if small enough.

---

## 12. Definition of Done

FNO integration is complete when all items below are true:

### Service

- `fno-service/` exists and is importable with `PYTHONPATH=fno-service/src`.
- `GET /health` works.
- `GET /ready` verifies checkpoint/model readiness.
- `POST /predict` returns backend-compatible JSON.
- Missing checkpoint behavior is explicit and tested.

### Model

- FNO model implementation is real PyTorch code, not a mock.
- Spectral convolution layer is covered by tests.
- Forward pass works on CPU.
- CUDA is supported when available.
- Model input/output channels are config-driven.

### Data

- FNO can train on regular grid tensors derived from the same COMSOL/PINN-compatible data source.
- Dataset metadata and channel names are validated.
- Grid masks are used or at least preserved.
- A tiny synthetic dataset exists for tests.

### Training

- `scripts/train_fno.py` runs from config.
- Smoke training completes on CPU.
- `best_model.pth`, `model.pth`, `metrics.json`, `training_config.json`, and metadata are saved.
- Validation loop exists.

### Backend

- Backend `model=fno` routes to `fno-service`, not default mock.
- Backend normalizer accepts FNO response.
- Existing PINN and MGN routes still work.
- Backend tests include FNO routing/response cases.

### Docker

- `docker compose config --quiet` passes.
- `docker compose up fno-service` starts the service.
- Full stack starts with backend depending on `fno-service`.
- `.env.example` documents FNO variables.

### Tests/CI

- `python -m pytest` includes FNO tests.
- CI has a dedicated `fno-tests` job.
- CI does not perform heavy training.
- Docker Compose config check includes the new service.

### Documentation

- `docs/FNO_INTEGRATION_ROADMAP.md` exists.
- `docs/FNO_SERVICE.md` exists.
- `docs/FNO_TRAINING.md` exists.
- `docs/FNO_API_CONTRACT.md` exists.
- `docs/FNO_DATASET_FORMAT.md` exists.
- README and model card no longer describe FNO as only a mock after implementation.

---

## Recommended First Codex Task

Use this as the first implementation prompt after the roadmap is accepted:

```text
Follow docs/FNO_INTEGRATION_ROADMAP.md.
Implement only Phase 2 and the minimal Phase 8 tests for the skeleton.
Do not implement the FNO neural network yet.
Create fno-service with Dockerfile, requirements, config loader, FastAPI app, /health, /ready, /predict placeholder that returns 503 CHECKPOINT_NOT_READY when no checkpoint exists, and tests for these endpoints.
Update docker-compose.yml and .env.example to add fno-service, but keep mock-fno under an optional demo-mocks profile.
Run docker compose config --quiet and FNO service tests.
Return a summary of changed files and any deviations from the roadmap.
```
