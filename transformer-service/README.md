# Transformer Inference Service

Autoregressive Transformer neural operator for thermoelastic-wave field prediction. Parallel to `pinn-service/`.

## Architecture

OFormer baseline (vanilla multi-head attention):

- **Encoder**: input tokens `(N, 16)` carrying coordinates, current-state fields and material properties; `L_enc` self-attention layers.
- **Decoder**: query tokens `(N, 3)` carrying target coordinates; `L_dec` cross-attention layers over the encoder output.
- **Head**: `Linear(d_model, 4)` predicting `(T, u, v, w)` at each query coordinate.

Training task: autoregressive time-stepping — predict the next state `(T, u, v, w)` given the current state.

## Supported COMSOL exports

Same format as `pinn-service`, plus standalone parsing tolerant of node-count mismatch between files (handled via coordinate intersection):

- `data_materials.csv`
- `data_temperature.csv`
- `data_displacement.csv` (may have fewer nodes — Dirichlet boundary excluded)
- `data_stress_1.csv`
- `data_stress_2.csv`
- `data_stress_3.csv`

The pilot dataset is `/Users/temporary/unik/sandstone experiment ROD/` with 4448/4743 node mismatch resolved at preprocessing.

## Data preparation

Strict 2D batch conversion from `rod_experiments_2d`:

```bash
PYTHONPATH=transformer-service/src python transformer-service/scripts/build_2d_transformer_datasets.py \
  --input-root pinn-service/artifacts/rod_experiments_2d \
  --output-root transformer-service/artifacts/datasets_2d
```

This writes per-rock pair datasets such as:

```text
transformer-service/artifacts/datasets_2d/granite_transformer_2d
transformer-service/artifacts/datasets_2d/limestone_transformer_2d
transformer-service/artifacts/datasets_2d/sandstone_transformer_2d
transformer-service/artifacts/datasets_2d/basalt_transformer_2d
```

and:

```text
transformer-service/artifacts/datasets_2d/manifest.json
```

```bash
PYTHONPATH=transformer-service/src python3 -m transformer_service.cli \
  --sandstone-dir "/Users/temporary/unik/sandstone experiment ROD" \
  --output-dir transformer-service/artifacts/sandstone \
  --build-pairs
```

Produces:

- `pairs.npz` — autoregressive `(state_t, state_{t+1})` tensors.
- `scalers.json` — per-channel mean/std for inputs and targets.
- `dataset_metadata.json` — node count, time grid, channel ordering.

## Training

```bash
PYTHONPATH=transformer-service/src python3 -m transformer_service.train \
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz \
  --output-dir transformer-service/artifacts/checkpoints/baseline_2d \
  --epochs 200 --device cpu
```

By default training subsamples 1024 input/query tokens per step (resolution-free style training). Override with `--n-tokens` or set to `0` / `-1` to use every node (slower).

Or the helper:

```bash
./transformer-service/train_baseline.sh
# Override knobs:
# EPOCHS=200 N_TOKENS=2048 D_MODEL=192 ./transformer-service/train_baseline.sh
```

Artifacts written to the output directory:

- `best_model.pth`
- `model.pth`
- `metrics.json`
- `training_config.json`
- `scalers.json`

## Inference service

The FastAPI app mirrors `pinn-service`:

```bash
PYTHONPATH=transformer-service/src \
TRANSFORMER_CHECKPOINT_PATH=transformer-service/artifacts/checkpoints/baseline_2d \
python3 -m uvicorn transformer_service.service_app:app --host 0.0.0.0 --port 9004
```

Endpoints:

- `GET /health`
- `GET /ready`
- `POST /predict`

## Current scope

Current practical scope:

- only supervised data MSE on `(T, u, v, w)`
- strict 2D training is now supported through `rod_experiments_2d` batch conversion
- no physics-informed residuals yet
- wired into `docker-compose.yml` and backend routing

These are next-iteration tasks.
