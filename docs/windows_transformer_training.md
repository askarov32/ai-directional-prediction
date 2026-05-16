# Windows Transformer Training

This guide shows how to prepare data and train `transformer-service` on Windows with PowerShell.

It is written for copy-paste usage.

## 1. Activate the virtual environment

If the repository already uses `.venv-pinn`:

```powershell
.\.venv-pinn\Scripts\Activate.ps1
```

If `python` is not recognized later, replace `python` with `py` in the commands below.

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r transformer-service/requirements.txt
```

If you also want plotting/report scripts later:

```powershell
pip install matplotlib pandas
```

## 3. Optional: install CUDA-enabled PyTorch

Use this only if:

- your Windows laptop has an NVIDIA GPU;
- CUDA drivers are installed;
- you want training with `--device cuda`.

Example for CUDA 12.4 wheels:

```powershell
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Quick check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, you can use `--device cuda`.
If it prints `False`, train with `--device cpu`.

## 4. Prepare the transformer dataset from raw CSV files or strict 2D artifacts

If you already built:

```text
pinn-service/artifacts/rod_experiments_2d/
```

then the preferred strict 2D path is to batch-convert those structured datasets into Transformer pair datasets.

Command:

```powershell
$env:PYTHONPATH="transformer-service/src"

python transformer-service/scripts/build_2d_transformer_datasets.py `
  --input-root pinn-service/artifacts/rod_experiments_2d `
  --output-root transformer-service/artifacts/datasets_2d
```

This creates, for example:

```text
transformer-service/artifacts/datasets_2d/granite_transformer_2d
transformer-service/artifacts/datasets_2d/limestone_transformer_2d
transformer-service/artifacts/datasets_2d/sandstone_transformer_2d
transformer-service/artifacts/datasets_2d/basalt_transformer_2d
transformer-service/artifacts/datasets_2d/manifest.json
```

For the new strict 2D baseline, prefer a dataset like:

```text
transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz
```

If you want the older one-off raw CSV path instead, it still works:

The service expects the COMSOL-style CSV directory described in [transformer-service/README.md](/Users/askarovi/Documents/New%20project/transformer-service/README.md).

Example if your sandstone raw files are here:

```text
data/sandstone
```

Run:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.cli `
  --sandstone-dir data/sandstone `
  --output-dir transformer-service/artifacts/sandstone `
  --build-pairs
```

Expected outputs:

```text
transformer-service/artifacts/sandstone/pairs.npz
transformer-service/artifacts/sandstone/scalers.json
transformer-service/artifacts/sandstone/dataset_metadata.json
```

## 5. Smoke training run

Before a long run, do one short epoch:

CPU:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.train `
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz `
  --output-dir transformer-service/artifacts/checkpoints/smoke `
  --epochs 1 `
  --device cpu
```

CUDA:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.train `
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz `
  --output-dir transformer-service/artifacts/checkpoints/smoke `
  --epochs 1 `
  --device cuda
```

## 6. Baseline training run

CPU:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.train `
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz `
  --output-dir transformer-service/artifacts/checkpoints/baseline_2d `
  --epochs 200 `
  --device cpu
```

CUDA:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.train `
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz `
  --output-dir transformer-service/artifacts/checkpoints/baseline_2d `
  --epochs 200 `
  --device cuda
```

## 7. Control the token budget

By default the training loop subsamples `1024` tokens per step.

You can increase this:

```powershell
$env:PYTHONPATH="transformer-service/src"

python -m transformer_service.train `
  --dataset transformer-service/artifacts/datasets_2d/limestone_transformer_2d/pairs.npz `
  --output-dir transformer-service/artifacts/checkpoints/baseline_2d `
  --epochs 200 `
  --n-tokens 2048 `
  --device cuda
```

Use all nodes:

```powershell
--n-tokens 0
```

This is slower and uses more memory.

## 8. Training artifacts

The output directory contains:

```text
best_model.pth
model.pth
metrics.json
training_config.json
scalers.json
```

Typical strict 2D baseline path:

```text
transformer-service/artifacts/checkpoints/baseline_2d
```

## 9. Connect the trained checkpoint to the service

In `.env`, point the service to the trained checkpoint directory:

```env
TRANSFORMER_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline_2d
TRANSFORMER_DEVICE=cpu
```

If Docker GPU is actually configured and working, you may use:

```env
TRANSFORMER_DEVICE=cuda
```

Then restart the service:

```powershell
docker compose up -d --build transformer-service
```

## 10. Verify readiness

```powershell
curl http://localhost:9004/ready
```

Expected signs of success:

- `ready: true`
- a resolved path to `best_model.pth`
- no checkpoint load error

## 11. Recommended workflow

1. Build `pairs.npz`
2. Run a 1-epoch smoke training
3. Run the real baseline training
4. Restart `transformer-service`
5. Check `/ready`
6. Use it in the comparison pipeline

## 12. Common issues

### `python` is not recognized

Use:

```powershell
py -m transformer_service.cli ...
```

or

```powershell
py -m transformer_service.train ...
```

### `No module named transformer_service`

Set:

```powershell
$env:PYTHONPATH="transformer-service/src"
```

in the same PowerShell session before running the commands.

### `torch.cuda.is_available()` is `False`

Train on CPU or reinstall CUDA-enabled PyTorch.

### The service still loads the old checkpoint

Check `.env`:

```env
TRANSFORMER_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline_2d
```

Then rebuild/restart:

```powershell
docker compose up -d --build transformer-service
```
