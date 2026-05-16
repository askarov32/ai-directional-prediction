# Windows PINN Training Guide

This guide is a copy-paste oriented setup for training the PINN model on a Windows machine.

It assumes:

- you are in the project root;
- you use `PowerShell`;
- you want to train the four-rock PINN stack from the prepared rod experiment dataset;
- you prefer GPU training when CUDA is available.
- your raw CSV files are stored under `data/granite`, `data/limestone`, `data/sandstone`, and `data/basalt`.

This guide now supports two architecture families:

- `mlp`: the original baseline PINN;
- `res_split`: the improved residual PINN with separate coordinate/material encoders and split thermal/displacement heads.

Public model contract is unchanged:

```text
input  = [x, y, z, t, E, nu, rho, alpha, k, Cp]
output = [T, u, v, w]
```

## 0. Expected Raw Data Layout

The project now expects the raw exports in the repository `data/` folder:

```text
data/
  granite/
    data_materials.csv
    data_temperature.csv
    data_displacement.csv
    data_stress_1.csv
    data_stress_2.csv
    data_stress_3.csv
    data_strain.csv
    granite_mesh.csv
  limestone/
    ...
  sandstone/
    ...
  basalt/
    ...
```

If you need to rebuild the processed PINN datasets from these raw CSV files, run the full preparation block in section 6.

```powershell
$env:PYTHONPATH="pinn-service/src"
python pinn-service/scripts/build_rod_experiments.py --raw-root data --output-dir pinn-service/artifacts/rod_experiments
```

Project recommendation:

- Python `3.11`
- virtual environment `.venv-pinn`
- PyTorch `2.6.0`

For official PyTorch install guidance, see:

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)

## 1. Open PowerShell In The Project Root

Example:

```powershell
cd "C:\Users\your-user\Documents\ai-directional-prediction"
```

## 2. Create And Activate The Virtual Environment

```powershell
py -3.11 -m venv .venv-pinn
.\.venv-pinn\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-pinn\Scripts\Activate.ps1
```

## 3. Install Base Project Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r pinn-service/requirements.txt
```

## 4. Replace CPU Torch With CUDA Torch

On Windows, plain `pip install -r pinn-service/requirements.txt` may leave you with a CPU-only PyTorch build.

For this project, reinstall the pinned PyTorch version with an official CUDA wheel.

Recommended command for `torch==2.6.0` with CUDA `12.4`:

```powershell
pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

If your machine needs another official PyTorch CUDA wheel, choose it from:

- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)

## 5. Verify That CUDA Is Really Visible

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected result:

- `cuda_available True`

If it prints `False`, do not start the long training run yet.

## 6. Prepare The PINN Datasets

Run this before `--dry-run` or full training on a fresh Windows machine.

This block:

- builds processed per-rock artifacts from `data/granite`, `data/limestone`, `data/sandstone`, and `data/basalt`;
- creates `training_samples_all_rocks.npz`;
- creates deterministic `train_samples.npz` and `val_samples.npz`;
- creates the data quality report;
- creates the initial loss-scale report used by normalized training.

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/build_rod_experiments.py `
  --raw-root data `
  --output-dir pinn-service/artifacts/rod_experiments

python pinn-service/scripts/create_train_val_split.py `
  --dataset pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz `
  --metadata pinn-service/artifacts/rod_experiments/training_samples_all_rocks_metadata.json `
  --output-dir pinn-service/artifacts/rod_experiments/splits `
  --val-fraction 0.1 `
  --seed 42

python pinn-service/scripts/generate_data_quality_report.py `
  --manifest pinn-service/artifacts/rod_experiments/manifest.json `
  --output-dir pinn-service/artifacts/rod_experiments/reports

python pinn-service/scripts/estimate_loss_scales.py `
  --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz `
  --output-dir pinn-service/artifacts/rod_experiments/reports `
  --sample-limit 8192 `
  --batch-size 512 `
  --device cuda
```

Expected files after this block:

- `pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz`
- `pinn-service/artifacts/rod_experiments/splits/train_samples.npz`
- `pinn-service/artifacts/rod_experiments/splits/val_samples.npz`
- `pinn-service/artifacts/rod_experiments/reports/data_quality_report.html`
- `pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json`

## 7. Optional Dry Run

This validates the dataset paths, validation split, loss-scale report, and resolved config without launching training.

Baseline dry run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --architecture mlp `
  --hidden-dim 192 `
  --depth 6 `
  --epochs 1 `
  --batch-size 128 `
  --sample-limit 256 `
  --validation-sample-limit 128 `
  --device cuda `
  --dry-run
```

Improved `res_split` dry run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --activation tanh `
  --epochs 1 `
  --batch-size 128 `
  --sample-limit 256 `
  --validation-sample-limit 128 `
  --device cuda `
  --dry-run
```

Improved `res_split` + Fourier dry run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --activation tanh `
  --use-fourier-features `
  --fourier-num-frequencies 6 `
  --fourier-scale 1.0 `
  --epochs 1 `
  --batch-size 128 `
  --sample-limit 256 `
  --validation-sample-limit 128 `
  --device cuda `
  --dry-run
```

## 8. Main Training Command For GPU

### 8.1 Baseline MLP

This is the safe baseline command:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_mlp_192x6 `
  --architecture mlp `
  --hidden-dim 192 `
  --depth 6 `
  --activation tanh `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```

### 8.2 Improved ResSplit PINN

This is the recommended improved architecture command:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_res_split `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --activation tanh `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```

### 8.3 Improved ResSplit PINN With Fourier Features

Use this only as a controlled experiment:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_res_split_fourier `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --activation tanh `
  --use-fourier-features `
  --fourier-num-frequencies 6 `
  --fourier-scale 1.0 `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```

### 8.4 Tapered MLP Baseline

This is useful if you want a stronger non-residual baseline:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_mlp_tapered `
  --architecture mlp `
  --mlp-layer-dims 256,256,192,192,128,128 `
  --activation tanh `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```

This script automatically uses:

- `pinn-service/artifacts/rod_experiments/splits/train_samples.npz`
- `pinn-service/artifacts/rod_experiments/splits/val_samples.npz`
- `pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json`

It also automatically:

- enables normalized loss balancing;
- writes `model.pth` and `best_model.pth`;
- saves `metrics.json`, `metrics.csv`, `training_config.json`, `scalers.json`;
- generates the HTML training report after training.

## 9. If GPU Memory Is Not Enough

Use a smaller batch size:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_res_split `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --epochs 2000 `
  --batch-size 4096 `
  --validation-batch-size 4096 `
  --device cuda
```

If needed, go lower:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_res_split `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --epochs 2000 `
  --batch-size 2048 `
  --validation-batch-size 2048 `
  --device cuda
```

## 10. CPU Fallback Command

If CUDA is unavailable, training can still run on CPU, but it will be much slower:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_cpu `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --epochs 2000 `
  --batch-size 2048 `
  --validation-batch-size 2048 `
  --device cpu
```

## 11. Quick Smoke Training

If you want to make sure the training loop works before the long run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir "$env:TEMP\pinn-training-smoke" `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 2 `
  --epochs 1 `
  --batch-size 64 `
  --sample-limit 128 `
  --validation-sample-limit 64 `
  --device cuda
```

## 12. Where To Look After Training

Main outputs:

```text
pinn-service/artifacts/checkpoints/<your-selected-run>/
```

Important files:

- `best_model.pth`
- `model.pth`
- `metrics.json`
- `metrics.csv`
- `training_config.json`
- `scalers.json`

Training report:

```text
pinn-service/artifacts/checkpoints/<your-selected-run>/report/training_report.html
```

Recommended output directories:

- `pinn-service/artifacts/checkpoints/rod_all_rocks_mlp_192x6`
- `pinn-service/artifacts/checkpoints/rod_all_rocks_mlp_tapered`
- `pinn-service/artifacts/checkpoints/rod_all_rocks_res_split`
- `pinn-service/artifacts/checkpoints/rod_all_rocks_res_split_fourier`

## 13. Common Problems

### `train_samples.npz` was not found

This means the train/validation split has not been created yet.

Run section 6 first:

```powershell
$env:PYTHONPATH="pinn-service/src"
python pinn-service/scripts/build_rod_experiments.py --raw-root data --output-dir pinn-service/artifacts/rod_experiments
python pinn-service/scripts/create_train_val_split.py --dataset pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz --metadata pinn-service/artifacts/rod_experiments/training_samples_all_rocks_metadata.json --output-dir pinn-service/artifacts/rod_experiments/splits --val-fraction 0.1 --seed 42
python pinn-service/scripts/estimate_loss_scales.py --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz --output-dir pinn-service/artifacts/rod_experiments/reports --sample-limit 8192 --batch-size 512 --device cuda
```

### `torch.cuda.is_available()` is `False`

Usually this means one of these:

- CPU-only PyTorch wheel is installed;
- NVIDIA driver is missing or outdated;
- the selected PyTorch CUDA wheel does not match the system well enough;
- you are not actually on an NVIDIA GPU machine.

Recommended recovery:

1. remove torch packages;
2. reinstall the official CUDA wheel;
3. rerun the CUDA check command.

Commands:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### Out Of Memory

Reduce:

- `--batch-size`
- `--validation-batch-size`

The safest first change is from `8192` to `4096`.

### Very Slow Training

Check that:

- `torch.cuda.is_available()` is `True`;
- `--device cuda` is used;
- batch size is not too small;
- no other heavy GPU workload is running.

### I am not sure which architecture to train first

Recommended order:

1. `mlp` baseline:
   - `--architecture mlp --hidden-dim 192 --depth 6`
2. `res_split`:
   - `--architecture res_split --hidden-dim 192 --num-blocks 4`
3. optional Fourier experiment:
   - `--architecture res_split --use-fourier-features`

If you only want one serious run first, use:

```text
--architecture res_split --hidden-dim 192 --num-blocks 4 --activation tanh
```

## 14. Minimal Copy-Paste Block

If you just want the shortest possible setup, use this:

```powershell
cd "C:\Users\your-user\Documents\ai-directional-prediction"
py -3.11 -m venv .venv-pinn
.\.venv-pinn\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r pinn-service/requirements.txt
pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$env:PYTHONPATH="pinn-service/src"
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python pinn-service/scripts/build_rod_experiments.py --raw-root data --output-dir pinn-service/artifacts/rod_experiments
python pinn-service/scripts/create_train_val_split.py --dataset pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz --metadata pinn-service/artifacts/rod_experiments/training_samples_all_rocks_metadata.json --output-dir pinn-service/artifacts/rod_experiments/splits --val-fraction 0.1 --seed 42
python pinn-service/scripts/generate_data_quality_report.py --manifest pinn-service/artifacts/rod_experiments/manifest.json --output-dir pinn-service/artifacts/rod_experiments/reports
python pinn-service/scripts/estimate_loss_scales.py --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz --output-dir pinn-service/artifacts/rod_experiments/reports --sample-limit 8192 --batch-size 512 --device cuda
python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_res_split `
  --architecture res_split `
  --hidden-dim 192 `
  --num-blocks 4 `
  --activation tanh `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```
