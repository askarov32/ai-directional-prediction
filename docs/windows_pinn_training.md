# Windows PINN Training Guide

This guide is a copy-paste oriented setup for training the PINN model on a Windows machine.

It assumes:

- you are in the project root;
- you use `PowerShell`;
- you want to train the four-rock PINN baseline from the prepared rod experiment dataset;
- you prefer GPU training when CUDA is available.

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

## 6. Optional Dry Run

This validates the dataset paths, validation split, loss-scale report, and resolved config without launching training.

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --epochs 1 `
  --batch-size 128 `
  --sample-limit 256 `
  --validation-sample-limit 128 `
  --device cuda `
  --dry-run
```

## 7. Main Training Command For GPU

This is the main copy-paste command for the full four-rock PINN run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_2000 `
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

## 8. If GPU Memory Is Not Enough

Use a smaller batch size:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_2000 `
  --epochs 2000 `
  --batch-size 4096 `
  --validation-batch-size 4096 `
  --device cuda
```

If needed, go lower:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_2000 `
  --epochs 2000 `
  --batch-size 2048 `
  --validation-batch-size 2048 `
  --device cuda
```

## 9. CPU Fallback Command

If CUDA is unavailable, training can still run on CPU, but it will be much slower:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_cpu `
  --epochs 2000 `
  --batch-size 2048 `
  --validation-batch-size 2048 `
  --device cpu
```

## 10. Quick Smoke Training

If you want to make sure the training loop works before the long run:

```powershell
$env:PYTHONPATH="pinn-service/src"

python pinn-service/scripts/run_training_experiment.py `
  --output-dir "$env:TEMP\pinn-training-smoke" `
  --epochs 1 `
  --batch-size 64 `
  --sample-limit 128 `
  --validation-sample-limit 64 `
  --device cuda
```

## 11. Where To Look After Training

Main outputs:

```text
pinn-service/artifacts/checkpoints/rod_all_rocks_2000/
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
pinn-service/artifacts/checkpoints/rod_all_rocks_2000/report/training_report.html
```

## 12. Common Problems

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

## 13. Minimal Copy-Paste Block

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
python pinn-service/scripts/run_training_experiment.py `
  --output-dir pinn-service/artifacts/checkpoints/rod_all_rocks_2000 `
  --epochs 2000 `
  --batch-size 8192 `
  --validation-batch-size 8192 `
  --device cuda
```
