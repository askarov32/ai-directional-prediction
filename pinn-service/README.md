# PINN Data Preparation

This folder contains the standalone PINN data, training, and inference stack:

- a robust parser for COMSOL CSV exports
- a validator that checks cross-file consistency
- a dataset builder that emits unified `.npz` artifacts for PINN research and training
- a first hybrid PINN trainer that can produce a checkpoint from the prepared data
- a FastAPI inference service that loads a real checkpoint when available

## Supported COMSOL exports

The current pipeline expects the six exports you shared:

- `data_materials.csv`
- `data_temperature.csv`
- `data_displacement.csv`
- `data_stress_1.csv`
- `data_stress_2.csv`
- `data_stress_3.csv`

## Output artifacts

The CLI writes:

- `structured_dataset.npz`
- `dataset_metadata.json`
- optional `training_samples.npz`

`structured_dataset.npz` stores field tensors grouped by physics domain.

`training_samples.npz` stores flattened `inputs` and `targets` that can be used as a starting point for supervised or hybrid PINN experiments.

## Run

From the project root:

```bash
python3 -m venv .venv-pinn
source .venv-pinn/bin/activate
pip install -r pinn-service/requirements.txt
PYTHONPATH=pinn-service/src python3 -m pinn_service.cli \
  --materials /path/to/data_materials.csv \
  --temperature /path/to/data_temperature.csv \
  --displacement /path/to/data_displacement.csv \
  --stress1 /path/to/data_stress_1.csv \
  --stress2 /path/to/data_stress_2.csv \
  --stress3 /path/to/data_stress_3.csv \
  --output-dir pinn-service/artifacts/demo \
  --build-training-matrix
```

## Train The First PINN Baseline

After building `training_samples.npz`, train the first hybrid baseline:

```bash
PYTHONPATH=pinn-service/src python3 -m pinn_service.train \
  --dataset pinn-service/artifacts/demo/training_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/baseline \
  --epochs 25 \
  --batch-size 4096 \
  --device cpu
```

Artifacts:

- `model.pth`
- `best_model.pth`
- `metrics.json`
- `training_config.json`
- `scalers.json`

For a stronger reusable baseline, use the helper script:

```bash
./pinn-service/train_baseline.sh
```

Default baseline script settings:

- `epochs=8`
- `batch_size=8192`
- `sample_limit=120000`
- `device=cpu`

Override them with environment variables:

```bash
EPOCHS=2000 BATCH_SIZE=8192 SAMPLE_LIMIT=120000 DEVICE=cpu ./pinn-service/train_baseline.sh
```

## Run The Inference Service

The inference service expects a trained checkpoint.

```bash
pip install -r pinn-service/requirements.txt
PYTHONPATH=pinn-service/src \
PINN_CHECKPOINT_PATH=pinn-service/artifacts/checkpoints/baseline \
python3 -m uvicorn pinn_service.service_app:app --host 0.0.0.0 --port 9003
```

Endpoints:

- `GET /health`
- `GET /ready`
- `POST /predict`

If the checkpoint is missing:

- `GET /health` returns `ready: false`
- `GET /ready` returns HTTP `503`
- `POST /predict` returns `503 CHECKPOINT_NOT_READY`

If `PINN_CHECKPOINT_PATH` points to a directory, the service auto-picks:

1. `best_model.pth`
2. `model.pth`
3. the first available `*.pth`

## Current PINN Strategy

This first version is a pragmatic hybrid PINN baseline:

- input: `x, y, z, t, E, nu, rho, alpha, k, Cp`
- model output: `T, u, v, w`
- supervised loss on COMSOL reference fields
- velocity-consistency loss using `ut, vt, wt`
- thermal residual regularization using a diffusion-style residual

This is intentionally a first trainable step, not yet the final fully coupled thermoelastic PINN formulation.

## Inference Readiness And Diagnostics

On startup, the service:

1. resolves `PINN_CHECKPOINT_PATH`;
2. loads `best_model.pth` or `model.pth`;
3. verifies feature metadata alignment;
4. runs deterministic smoke inference;
5. checks output shape and finite values.

`GET /ready` exposes:

- checkpoint path;
- resolved checkpoint file;
- device;
- active input feature names;
- output feature names;
- best training loss stored in checkpoint;
- smoke-check status.

Prediction responses keep the original flat fields required by the backend normalizer and also include:

- `model_outputs`: raw neural output feature names and values;
- `postprocessed_prediction`: final direction/summary fields;
- `diagnostics`: checkpoint and postprocessing metadata.

The final directional prediction combines neural outputs with geometry/material postprocessing. This is explicit by design for MVP transparency.

## Current assumptions

- the six files belong to one consistent COMSOL simulation family
- all files share the same node order and the same time grid
- the current dataset is 3D and single-scenario
- materials appear effectively time-invariant and are reduced to static node-wise fields

## Training matrix contract

Inputs:

- `x`
- `y`
- `z`
- `t`
- `youngs_modulus`
- `poissons_ratio`
- `density`
- `thermal_expansion`
- `thermal_conductivity`
- `heat_capacity`

Targets:

- `temperature_k`
- `disp_x`
- `disp_y`
- `disp_z`
- `vel_x`
- `vel_y`
- `vel_z`
- `von_mises`
- `stress_x`
- `stress_y`
- `stress_z`
- `stress_xy`
- `stress_yz`
- `stress_xz`
- `strain_x`
- `strain_y`
- `strain_z`
