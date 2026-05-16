# FNO Service

This is the service for the Fourier Neural Operator route.

Current phase:

```text
Phase 6: scale-aware FNO2d baseline
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
- checkpoint-based FNO inference using local dataset conditioning;
- backend-compatible `/predict` response from a trained `best_model.pth` or `model.pth`.
- channel-wise input/target normalization for newly trained checkpoints;
- checkpoint inference that denormalizes predicted channels before computing physical metrics;
- 2D-specific metric calculation and explicit scale warnings.

Not implemented yet:

- field inference.
- standalone evaluation CLI;
- dedicated `run_inference.py` helper script.
- explicit thermoelastic PDE residuals. FNO is a neural operator baseline, not a PINN.

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

If `FNO_ALLOW_FALLBACK=true`, `/ready` can report ready and `/predict` returns a deterministic placeholder response when no checkpoint is available. This is only for local stack wiring and quick demos.

If a checkpoint directory contains `best_model.pth` or `model.pth`, the service now loads it and runs real `FNO2d` inference. The request itself is still lightweight: the backend sends medium/scenario/source/probe/domain, and the FNO service combines that request with its local grid dataset to build the model input.

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

For thesis experiments, use clean 2D requests:

```text
domain.type = rect_2d
domain.size.lz = 0.0
domain.resolution.nz = 1
source.z = 0.0
probe.z = 0.0
source.direction[2] = 0.0
```

Clean 2D inference reports:

```text
diagnostics.effective_domain_type = rect_2d
diagnostics.domain_adaptation = none
```

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

The current objective is intentionally simple and stable for the MVP:

```text
loss = MSE(predicted_next_fields, target_next_fields)
     + 0.01 * relative_l2(predicted_next_fields, target_next_fields)
```

Current targets are:

```text
temperature_k, disp_x, disp_y, disp_z
```

### Channel-wise Normalization

Newly trained FNO checkpoints store channel-wise normalization metadata in `channel_metadata.normalization`:

```text
input.channel_names
input.mean / input.std / input.min / input.max / input.units
target.channel_names
target.mean / target.std / target.min / target.max / target.units
```

Training now standardizes every input and target channel separately. This is important because temperature values and displacement values can differ by many orders of magnitude.

Inference uses the same metadata:

```text
1. Build physical input channels.
2. Normalize input channels with checkpoint input statistics.
3. Run FNO2d.
4. Denormalize predicted target channels with checkpoint target statistics.
5. Compute physical metrics from denormalized fields.
```

Older checkpoints that do not contain normalization metadata are still loadable for compatibility, but `/predict` will include warnings:

```text
missing_input_normalization_metadata
missing_output_denormalization_metadata
```

Those older results should be interpreted as scale-unsafe until the FNO model is retrained with the updated training stack.

## Inference

Current inference path:

```text
backend request
  -> POST /predict on fno-service
  -> load checkpoint + local FNO grid dataset
  -> build conditioned input tensor
  -> run FNO2d
  -> summarize direction/travel time/field metrics
```

Metric definitions:

```text
max_displacement = max(sqrt(disp_x^2 + disp_y^2))
```

For `rect_2d`, `disp_z` is ignored and returned direction has:

```text
direction_z = 0.0
elevation_deg = 0.0
```

Temperature perturbation is computed as:

```text
max_temperature_perturbation = max(abs(temperature_k - reference_temperature_k))
```

`magnitude` is a bounded dimensionless comparative response score for frontend/model comparison. It is not a displacement unit.

The response diagnostics include:

```text
normalization_used
denormalization_used
warnings
metric_definitions
```

Prototype sanity warnings are emitted for suspicious scales:

```text
max_displacement > 1e2
max_temperature_perturbation > 1e4
magnitude > 1e6
```

These thresholds are not universal physical limits; they are safeguards for identifying unstable or scale-inconsistent prototype outputs.

Current limitations:

```text
- supports rect_2d only;
- requires FNO grid data with Z=1;
- uses local dataset conditioning instead of sending full grids through the backend;
- intended as an MVP baseline, not a final validated scientific model.
- does not explicitly enforce thermoelastic PDE residuals like the PINN service;
- existing pre-normalization checkpoints should be retrained before final thesis comparison.
```
