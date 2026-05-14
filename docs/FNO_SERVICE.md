# FNO Service

## Purpose

`fno-service` is the dedicated `FNO` route behind the backend `PredictionRouter`.

It is responsible for:

- loading a trained `FNO2d` checkpoint;
- loading a local regular-grid dataset;
- building a conditioned model input from backend request data plus local grid tensors;
- running inference;
- returning a backend-compatible response shape.

## Current Status

Current implementation:

- real FastAPI service;
- `GET /health`;
- `GET /ready`;
- `POST /predict`;
- `FNO2d` checkpoint loading;
- CPU/CUDA device resolution;
- deterministic fallback mode when checkpoint is absent and `FNO_ALLOW_FALLBACK=true`.

Current limitations:

- current model is `FNO2d`, not `FNO3d`;
- current inference supports `rect_2d` only;
- current regular-grid artifacts must have `Z=1`;
- request does not carry full grid tensors through the backend.

## Runtime Environment

Important variables:

```text
FNO_CHECKPOINT_PATH
FNO_CONFIG_PATH
FNO_DATASET_PATH
FNO_DEVICE
FNO_LOG_LEVEL
FNO_ALLOW_FALLBACK
```

Typical Docker defaults:

```text
FNO_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline
FNO_CONFIG_PATH=/app/configs/inference.yaml
FNO_DATASET_PATH=/app/artifacts/datasets/sandstone_fno
FNO_DEVICE=cpu
FNO_ALLOW_FALLBACK=true
```

## Device Behavior

- if `FNO_DEVICE=cpu`, inference runs on CPU;
- if `FNO_DEVICE=cuda` and CUDA is available in the installed PyTorch build, inference runs on GPU;
- if `FNO_DEVICE=cuda` is requested but CUDA is unavailable, the service falls back to CPU instead of crashing.

## Endpoints

```text
GET /health
GET /ready
POST /predict
```

`GET /health`:

- does not require a checkpoint;
- reports service state and configured paths.

`GET /ready`:

- reports `200` when checkpoint mode is ready or fallback mode is enabled;
- reports `503` when no checkpoint is present and fallback is disabled.

`POST /predict`:

- returns real checkpoint inference when a checkpoint is available;
- returns fallback output only when fallback mode is enabled;
- returns structured service errors for unsupported domains, model-load failures, or non-finite outputs.

## Response Shape

The service returns:

```json
{
  "prediction": {
    "direction_vector": [0.821, 0.571, 0.0],
    "azimuth_deg": 34.8,
    "elevation_deg": 0.0,
    "magnitude": 0.914,
    "wave_type": "fno_checkpoint_inference",
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

The backend normalizer ignores extra diagnostic fields and keeps the public frontend contract stable.
