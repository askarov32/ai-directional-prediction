# FNO Service

This is the service skeleton for the Fourier Neural Operator route.

Current phase:

```text
Phase 5: trainable FNO2d baseline
```

Implemented now:

- FastAPI app;
- `GET /health`;
- `GET /ready`;
- `POST /predict`;
- environment-based config;
- missing-checkpoint behavior;
- optional fallback response for local demo wiring.
- FNO grid dataset loader for `grid_dynamic.npy`, `grid_static.npy`, `grid_masks.npy`, and metadata;
- fallback converter from PINN `structured_dataset.npz` to regular FNO grid tensors.
- `SpectralConv2d`;
- `FNO2d` forward model.
- FNO training loop;
- train/validation split over time-step pairs;
- checkpoint, metrics, and channel metadata artifacts;
- `scripts/train_fno.py` CLI.

Not implemented yet:

- checkpoint loading;
- field inference.

## Runtime

```bash
PYTHONPATH=fno-service/src python -m uvicorn fno_service.api.main:app --host 0.0.0.0 --port 9002
```

## Environment

```text
FNO_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline
FNO_CONFIG_PATH=/app/configs/inference.yaml
FNO_DATASET_PATH=/app/artifacts/datasets/sandstone_fno
FNO_DEVICE=cpu
FNO_LOG_LEVEL=INFO
FNO_ALLOW_FALLBACK=false
```

If `FNO_ALLOW_FALLBACK=true`, `/ready` can report ready and `/predict` returns a deterministic placeholder response. This is only for local stack wiring while the real FNO model is not implemented yet.

## Dataset Preparation

Validate an existing universal formatter FNO directory:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/prepare_fno_dataset.py \
  --source mgn-service/datasets/sandstone_comsol_real/processed/fno \
  --output-dir fno-service/artifacts/datasets/sandstone_fno \
  --validate
```

Fallback conversion from a PINN structured dataset:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/prepare_fno_dataset.py \
  --pinn-structured pinn-service/artifacts/demo/structured_dataset.npz \
  --pinn-metadata pinn-service/artifacts/demo/dataset_metadata.json \
  --output-dir fno-service/artifacts/datasets/demo_fno \
  --grid-res 1 32 32 \
  --max-timesteps 128 \
  --validate
```

Expected FNO layout:

```text
grid_dynamic.npy       # [T,C,Z,Y,X]
grid_static.npy        # [S,Z,Y,X]
grid_masks.npy         # [M,Z,Y,X]
grid_coords.npy        # [3,Z,Y,X]
field_names.json
static_feature_names.json
mask_names.json
metadata.json
```

## Model

Current implemented model:

```text
FNO2d
```

Current layer stack:

```text
input_projection:  Conv2d(in_channels -> width, kernel_size=1)
FNO blocks:        depth x (SpectralConv2d(width -> width) + pointwise Conv2d(width -> width))
activation:        GELU
output_projection: Conv2d(width -> 2*width) + GELU + Conv2d(2*width -> out_channels)
```

Default-style parameters planned for training configs:

```text
width = 32
modes_x = 12
modes_y = 12
depth = 4
```

The model expects 2D tensors:

```text
input:  [batch, channels, height, width]
output: [batch, output_channels, height, width]
```

For 3D source data, the current MVP path can train/infer on `Z=1` slices produced by the FNO grid converter. Native `FNO3d` is intentionally left for a later phase after the 2D baseline is trained and validated.

## Training

Minimal smoke training on a prepared FNO grid directory:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/train_fno.py \
  --config fno-service/configs/train_fno.yaml \
  --dataset-path fno-service/artifacts/datasets/demo_fno \
  --output-dir fno-service/artifacts/checkpoints/demo_fno \
  --epochs 1 \
  --batch-size 1 \
  --width 8 \
  --modes-x 2 \
  --modes-y 2 \
  --depth 1 \
  --device cpu
```

Default training config:

```text
fno-service/configs/train_fno.yaml
```

Training artifacts:

```text
model.pth
best_model.pth
metrics.json
metrics.csv
training_config.json
dataset_metadata.json
channel_metadata.json
```

The Phase 5 objective is intentionally simple and stable for the MVP:

```text
loss = MSE(predicted_next_fields, target_next_fields)
     + 0.01 * relative_l2(predicted_next_fields, target_next_fields)
```

Current targets are:

```text
temperature_k, disp_x, disp_y, disp_z
```
