# PINN Data Preparation

This folder contains the standalone PINN data, training, and inference stack:

- a robust parser for COMSOL CSV exports
- a validator that checks cross-file consistency
- a dataset builder that emits unified `.npz` artifacts for PINN research and training
- a first hybrid PINN trainer that can produce a checkpoint from the prepared data
- a FastAPI inference service that loads a real checkpoint when available

For a Windows-specific copy-paste training flow, see:

- [Windows PINN Training Guide](../docs/windows_pinn_training.md)

## Supported COMSOL exports

The current pipeline expects the core COMSOL exports:

- `data_materials.csv`
- `data_temperature.csv`
- `data_displacement.csv`
- `data_stress_1.csv`
- `data_stress_2.csv`
- `data_stress_3.csv`

It can also consume optional experiment files:

- `data_strain.csv` for full normal and shear strain components
- `<rock>_mesh.csv` or `<rock>.mphtxt`, stored in metadata only

For the rod experiments, `data_displacement.csv` may have fewer exported rows than the other files. Use `--coordinate-policy intersection` to align all fields by common coordinates and record dropped duplicate rows in `dataset_metadata.json`.

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
  --strain /path/to/data_strain.csv \
  --rock-id granite \
  --experiment-id granite_rod \
  --coordinate-policy intersection \
  --output-dir pinn-service/artifacts/demo \
  --build-training-matrix
```

To build all four rod experiments from `~/Downloads`:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/build_rod_experiments.py \
  --raw-root ~/Downloads \
  --output-dir pinn-service/artifacts/rod_experiments
```

This writes one processed folder per rock, a shared `manifest.json`, and a combined `training_samples_all_rocks.npz` for multi-medium PINN training.

## Training Readiness Reports

Before long training runs, generate quality, split, and initial loss-scale diagnostics:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/generate_data_quality_report.py

PYTHONPATH=pinn-service/src python3 pinn-service/scripts/create_train_val_split.py \
  --val-fraction 0.1 \
  --seed 42

PYTHONPATH=pinn-service/src python3 pinn-service/scripts/estimate_loss_scales.py \
  --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz \
  --sample-limit 8192 \
  --batch-size 512 \
  --device cpu
```

Outputs are written under:

```text
pinn-service/artifacts/rod_experiments/reports/
pinn-service/artifacts/rod_experiments/splits/
```

The loss-scale report is meant to drive the new normalized training mode. It estimates raw component magnitudes at random initialization so we do not have to guess `wave` and `thermal` balancing for long runs.

## Train The First PINN Baseline

After building `training_samples.npz`, train the coupled thermoelastic PINN baseline:

```bash
PYTHONPATH=pinn-service/src python3 -m pinn_service.train \
  --dataset pinn-service/artifacts/demo/training_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/baseline \
  --epochs 25 \
  --batch-size 4096 \
  --device cpu \
  --wave-residual-weight 0.1 \
  --thermal-residual-weight 0.05 \
  --reference-temperature-k 293.15 \
  --max-grad-norm 1.0 \
  --physics-mode coupled_thermoelastic
```

For the four-rock rod dataset, use:

```bash
PYTHONPATH=pinn-service/src python3 -m pinn_service.train \
  --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz \
  --val-dataset pinn-service/artifacts/rod_experiments/splits/val_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_baseline \
  --epochs 2000 \
  --batch-size 8192 \
  --validation-batch-size 8192 \
  --device cpu \
  --wave-residual-weight 0.1 \
  --thermal-residual-weight 0.05 \
  --reference-temperature-k 293.15 \
  --loss-balance-mode normalize \
  --loss-scale-report pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json \
  --max-grad-norm 1.0 \
  --lr-scheduler-patience 20 \
  --lr-scheduler-factor 0.5 \
  --early-stopping-patience 60 \
  --early-stopping-min-delta 1e-4 \
  --physics-mode coupled_thermoelastic
```

Artifacts:

- `model.pth`
- `best_model.pth`
- `metrics.json`
- `metrics.csv`
- `training_config.json`
- `scalers.json`

When `--val-dataset` is provided, `best_model.pth` is selected by `val_total_loss`. Without validation data, it falls back to training `total_loss`. `model.pth` always stores the final epoch state.
`ReduceLROnPlateau` now tracks the same metric and lowers the learning rate when progress stalls. Early stopping uses that same target metric, so the checkpoint choice and stopping rule stay aligned.

## Recommended Long Training Command

For the current four-rock rod dataset, prefer the experiment runner. It uses the deterministic train/validation split, reads the initial loss-scale report, enables normalized loss balancing, writes checkpoints, and generates the HTML training report after training:

```bash
PYTHONPATH=pinn-service/src .venv-pinn/bin/python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_2000 \
  --epochs 2000 \
  --batch-size 8192 \
  --validation-batch-size 8192 \
  --device cuda
```

If CUDA is not available, use:

```bash
PYTHONPATH=pinn-service/src .venv-pinn/bin/python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_cpu \
  --epochs 2000 \
  --batch-size 2048 \
  --validation-batch-size 2048 \
  --device cpu
```

For a quick smoke check that does not touch the main checkpoint:

```bash
PYTHONPATH=pinn-service/src .venv-pinn/bin/python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir /tmp/pinn-training-smoke \
  --epochs 1 \
  --batch-size 64 \
  --sample-limit 128 \
  --validation-sample-limit 64 \
  --device cpu
```

For a stronger reusable baseline, use the helper script:

```bash
./pinn-service/train_baseline.sh
```

Default baseline script settings:

- `epochs=8`
- `batch_size=8192`
- `sample_limit=120000`
- `device=cpu`
- `supervised_weight=1.0`
- `velocity_weight=0.25`
- `wave_residual_weight=0.1`
- `thermal_residual_weight=0.05`
- `reference_temperature_k=293.15`
- `loss_balance_mode=fixed`
- `max_grad_norm=1.0`
- `min_learning_rate=1e-6`
- `lr_scheduler_patience=25`
- `lr_scheduler_factor=0.5`
- `physics_mode=coupled_thermoelastic`

Override them with environment variables:

```bash
EPOCHS=2000 BATCH_SIZE=8192 SAMPLE_LIMIT=120000 DEVICE=cpu \
VAL_DATASET_PATH=pinn-service/artifacts/rod_experiments/splits/val_samples.npz \
VALIDATION_BATCH_SIZE=8192 VALIDATION_SAMPLE_LIMIT=120000 \
WAVE_RESIDUAL_WEIGHT=0.1 THERMAL_RESIDUAL_WEIGHT=0.05 \
LOSS_BALANCE_MODE=normalize \
LOSS_SCALE_REPORT=pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json \
REFERENCE_TEMPERATURE_K=293.15 PHYSICS_MODE=coupled_thermoelastic \
MAX_GRAD_NORM=1.0 MIN_LEARNING_RATE=1e-6 \
LR_SCHEDULER_PATIENCE=20 LR_SCHEDULER_FACTOR=0.5 \
EARLY_STOPPING_PATIENCE=60 EARLY_STOPPING_MIN_DELTA=1e-4 \
./pinn-service/train_baseline.sh
```

If you already know your preferred scales, you can pass them directly instead of a report:

```bash
PYTHONPATH=pinn-service/src python3 -m pinn_service.train \
  --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_manual_scales \
  --epochs 2000 \
  --batch-size 8192 \
  --device cpu \
  --loss-balance-mode normalize \
  --supervised-loss-scale 1.0 \
  --velocity-loss-scale 95.69 \
  --wave-residual-loss-scale 3.93e13 \
  --thermal-residual-loss-scale 5.03e17
```

`metrics.csv` is written next to `metrics.json` for quick plotting. It includes per-epoch raw losses (`supervised_loss`, `velocity_consistency_loss`, `wave_residual_loss`, `thermal_residual_loss`), normalized losses (`normalized_supervised_loss`, `normalized_velocity_consistency_loss`, `normalized_wave_residual_loss`, `normalized_thermal_residual_loss`), `total_loss`, `grad_norm`, `learning_rate`, `epochs_without_improvement`, and `best_so_far`. `max_grad_norm` clips gradients before the optimizer step; set it to `0` to disable clipping.

With validation enabled, the CSV also includes `val_*` loss columns. Validation uses the training scalers, so it measures generalization on the same normalized feature space rather than fitting fresh statistics on the validation split. `metrics.json` also records whether training stopped early and how many epochs actually completed.

## Generate A Training Report

After training finishes, generate a compact HTML report with SVG charts:

```bash
python3 pinn-service/scripts/generate_training_report.py \
  --metrics-json pinn-service/artifacts/checkpoints/rod_all_rocks_baseline/metrics.json
```

Outputs are written by default to:

```text
pinn-service/artifacts/checkpoints/rod_all_rocks_baseline/report/
```

The report includes:

- `training_report.html`
- `training_report_summary.json`
- `total_loss.svg`
- `component_loss.svg`
- `normalized_loss.svg`
- `optimization.svg`

The script works with older checkpoints that only contain `metrics.json` and with newer checkpoints that also include `metrics.csv`.

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

This version trains a hybrid coupled thermoelastic PINN baseline:

- input: `x, y, z, t, E, nu, rho, alpha, k, Cp`
- model output: `T, u, v, w`
- supervised loss on COMSOL reference fields
- velocity-consistency loss using `ut, vt, wt`
- elastic wave residual from the divergence of the thermoelastic stress tensor
- coupled thermal residual with the `gamma * T0 * d(eps_kk)/dt` thermoelastic coupling term

Material parameters are treated as locally homogeneous pointwise features. The current loss does not take spatial derivatives of `E`, `nu`, `rho`, `alpha`, `k`, or `Cp`, because the current training matrix provides them as independent features rather than differentiable material fields.

The total training objective is:

```text
loss_total =
  supervised_weight * balanced(loss_supervised)
  + velocity_weight * balanced(loss_velocity)
  + wave_residual_weight * balanced(loss_wave)
  + thermal_residual_weight * balanced(loss_thermal)
```

Where:

- `balanced(loss) = loss` in `loss_balance_mode=fixed`
- `balanced(loss) = loss / component_scale` in `loss_balance_mode=normalize`

Typical normalized training flow:

1. generate `loss_scale_report.json`;
2. start training with `--loss-balance-mode normalize --loss-scale-report ...`;
3. keep interpretable high-level weights such as `wave_residual_weight=0.1` and `thermal_residual_weight=0.05`, while the component scales absorb the raw unit mismatch.

For backward compatibility, `--physics-mode simple_heat` keeps the older heat-equation residual and disables the wave residual contribution.

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
- rod experiment files can be aligned by common coordinates with `--coordinate-policy intersection`
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
- `strain_xy`
- `strain_yz`
- `strain_xz`
