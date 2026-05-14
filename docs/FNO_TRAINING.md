# FNO Training

## Training Scope

Current FNO training in this repository is an MVP `FNO2d` baseline.

It trains on regular-grid tensors with the next-step objective:

- input: dynamic fields at time `t` plus static/material/mask/coordinate/time channels;
- target: primary fields at time `t + 1`.

Current output targets:

```text
temperature_k
disp_x
disp_y
disp_z
```

## Dataset Preparation

Preferred path:

1. prepare or reuse regular-grid artifacts;
2. point `FNO_DATASET_PATH` to that directory.

Fallback local converter:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/prepare_fno_dataset.py \
  --pinn-structured pinn-service/artifacts/demo/structured_dataset.npz \
  --pinn-metadata pinn-service/artifacts/demo/dataset_metadata.json \
  --output-dir fno-service/artifacts/datasets/demo_fno \
  --grid-res 1 32 32 \
  --max-timesteps 128 \
  --validate
```

For the current `FNO2d` baseline, keep:

- `Z=1`;
- `rect_2d` inference assumptions.

## Training Command

Minimal smoke training:

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

## CUDA Usage

On a machine with CUDA-enabled PyTorch:

```bash
PYTHONPATH=fno-service/src python fno-service/scripts/train_fno.py \
  --config fno-service/configs/train_fno.yaml \
  --dataset-path fno-service/artifacts/datasets/demo_fno \
  --output-dir fno-service/artifacts/checkpoints/demo_fno_cuda \
  --epochs 10 \
  --batch-size 4 \
  --device cuda
```

Notes:

- `--device cuda` only helps if the installed PyTorch build includes CUDA support;
- if you are on macOS without CUDA, use `--device cpu`;
- the runtime inference service also supports `FNO_DEVICE=cuda` with CPU fallback when CUDA is unavailable.

## Current Objective

Current training loss:

```text
loss = MSE(predicted_next_fields, target_next_fields)
     + 0.01 * relative_l2(predicted_next_fields, target_next_fields)
```

This is intentionally simple for the MVP and should be extended later with stronger normalization, validation, and rollout-aware evaluation.

## Output Artifacts

Training writes:

```text
model.pth
best_model.pth
metrics.json
metrics.csv
training_config.json
dataset_metadata.json
channel_metadata.json
```

The inference service auto-prefers:

```text
best_model.pth
```

over:

```text
model.pth
```

when both are present.
