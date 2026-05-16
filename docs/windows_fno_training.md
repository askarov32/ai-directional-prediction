# Windows FNO Training Guide

This guide is a copy-paste oriented setup for training the current FNO baseline on a Windows machine.

It assumes:

- you are in the project root;
- you use `PowerShell`;
- you want to train the current `FNO2d` baseline;
- you prefer GPU training when CUDA is available;
- you understand that the current FNO path is a 2D regular-grid baseline with `Z=1`, not a full 3D production setup.

## What Is Ready Right Now

Yes, the repository is ready for FNO training in its current MVP form.

Current scope:

- model: `FNO2d`;
- training target: next-step field prediction;
- outputs: `temperature_k`, `disp_x`, `disp_y`, `disp_z`;
- input source: regular-grid tensors derived from the demo PINN structured dataset;
- inference assumption: `rect_2d`, `Z=1`;
- channel-wise normalization is now part of the FNO training/inference stack.

What this means in practice:

- you can already train a demo/baseline FNO checkpoint;
- you can use CPU or CUDA;
- after training, `fno-service` can load `best_model.pth` from the default baseline directory;
- newly trained checkpoints will include normalization metadata that inference uses for denormalization and metric calculation.

What this does **not** mean yet:

- it is not the same level of dataset/training pipeline maturity as the full multi-rock PINN stack;
- it is not yet a validated 3D scientific baseline;
- it is not yet a large four-rock training workflow like the current PINN experiment pipeline.

Important note:

- older FNO checkpoints created before the normalization update can still load, but they may return warnings such as `missing_input_normalization_metadata` and `missing_output_denormalization_metadata`;
- for thesis-quality comparison, retrain the FNO checkpoint with the current code before trusting the output scale.

## 1. Open PowerShell In The Project Root

Example:

```powershell
cd "C:\Users\your-user\Documents\ai-directional-prediction"
```

## 2. Create And Activate The Virtual Environment

You can reuse `.venv-pinn` because it already contains the needed Python stack, or create a separate environment. The fastest path is to reuse it.

```powershell
py -3.11 -m venv .venv-pinn
.\.venv-pinn\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-pinn\Scripts\Activate.ps1
```

## 3. Install FNO Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r fno-service/requirements.txt
```

## 4. Replace CPU Torch With CUDA Torch

On Windows, plain `pip install -r fno-service/requirements.txt` may leave you with a CPU-only PyTorch build.

For GPU training, reinstall the pinned PyTorch version with an official CUDA wheel.

Recommended command for `torch==2.6.0` with CUDA `12.4`:

```powershell
pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

If your machine needs another official CUDA wheel, choose it from:

- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)

## 5. Verify That CUDA Is Really Visible

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected result:

- `cuda_available True`

If it prints `False`, do not start the longer GPU training command yet. Use CPU first or fix the PyTorch install.

## 6. Prepare The FNO Dataset

The current easiest Windows path is to derive the FNO regular-grid dataset from an already built PINN `structured_dataset.npz` inside:

```text
pinn-service/artifacts/rod_experiments/<rock>/
```

First check which processed rocks already exist:

```powershell
Get-ChildItem pinn-service/artifacts/rod_experiments -Recurse -Filter structured_dataset.npz
```

Then choose one real existing path. For example, if `limestone` already exists, use it directly.

This creates:

- `grid_dynamic.npy`
- `grid_static.npy`
- `grid_masks.npy`
- `grid_coords.npy`
- metadata JSON files

The current FNO baseline expects a 2D grid:

```text
Z = 1
domain.type = rect_2d
```

Command:

Example for `limestone`:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/prepare_fno_dataset.py `
  --pinn-structured pinn-service/artifacts/rod_experiments/limestone/structured_dataset.npz `
  --pinn-metadata pinn-service/artifacts/rod_experiments/limestone/dataset_metadata.json `
  --output-dir fno-service/artifacts/datasets/limestone_fno `
  --grid-res 1 32 32 `
  --max-timesteps 64 `
  --validate
```

Important:

- keep the first grid dimension as `1`, because the current baseline expects `Z=1`;
- this is the current MVP path for `FNO2d`.
- if you want another rock, replace `limestone` with the rock folder that actually exists on your machine.

## 7. Optional Dry Run

This checks the resolved training config without starting training:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir "$env:TEMP\fno-dry-run" `
  --epochs 1 `
  --batch-size 1 `
  --width 8 `
  --modes-x 2 `
  --modes-y 2 `
  --depth 1 `
  --device cuda `
  --dry-run
```

If CUDA is not available yet, switch `--device cpu`.

## 8. Quick Smoke Training

Use this first before a longer run:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir "$env:TEMP\fno-training-smoke" `
  --epochs 1 `
  --batch-size 1 `
  --width 8 `
  --modes-x 2 `
  --modes-y 2 `
  --depth 1 `
  --device cuda
```

If needed, replace `cuda` with `cpu`.

## 9. Main GPU Training Command

This is the main copy-paste command for the current FNO baseline:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir fno-service/artifacts/checkpoints/baseline `
  --epochs 20 `
  --batch-size 4 `
  --learning-rate 0.001 `
  --weight-decay 0.000001 `
  --width 16 `
  --modes-x 6 `
  --modes-y 6 `
  --depth 2 `
  --device cuda
```

This is still an MVP baseline. Start modestly, inspect metrics, and only then scale epochs or width upward.

For a slightly stronger first real run on Windows GPU, this is a practical next step:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir fno-service/artifacts/checkpoints/baseline `
  --epochs 50 `
  --batch-size 4 `
  --learning-rate 0.001 `
  --weight-decay 0.000001 `
  --width 16 `
  --modes-x 6 `
  --modes-y 6 `
  --depth 2 `
  --device cuda
```

## 10. If GPU Memory Is Not Enough

Lower the batch size first:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir fno-service/artifacts/checkpoints/baseline `
  --epochs 20 `
  --batch-size 2 `
  --width 16 `
  --modes-x 6 `
  --modes-y 6 `
  --depth 2 `
  --device cuda
```

If needed, go lower again:

```powershell
--batch-size 1
```

## 11. CPU Fallback Command

If CUDA is unavailable, training can still run on CPU:

```powershell
$env:PYTHONPATH="fno-service/src"

python fno-service/scripts/train_fno.py `
  --config fno-service/configs/train_fno.yaml `
  --dataset-path fno-service/artifacts/datasets/limestone_fno `
  --output-dir fno-service/artifacts/checkpoints/baseline_cpu `
  --epochs 10 `
  --batch-size 1 `
  --width 8 `
  --modes-x 4 `
  --modes-y 4 `
  --depth 1 `
  --device cpu
```

## 12. One-Command Baseline Bootstrap

If you want the shortest path and you are comfortable with Git Bash or WSL:

```bash
DEVICE=cuda ./fno-service/train_baseline.sh
```

On pure PowerShell, it is safer to use the explicit Python commands from sections 6 and 9.

## 13. What Gets Saved Now

After training, check:

- `fno-service/artifacts/checkpoints/baseline/model.pth`
- `fno-service/artifacts/checkpoints/baseline/best_model.pth`
- `fno-service/artifacts/checkpoints/baseline/metrics.json`
- `fno-service/artifacts/checkpoints/baseline/metrics.csv`
- `fno-service/artifacts/checkpoints/baseline/training_config.json`
- `fno-service/artifacts/checkpoints/baseline/channel_metadata.json`

Now pay special attention to:

- `channel_metadata.json`

It should contain normalization metadata similar to:

```text
normalization.mode = channel_wise_standardization
normalization.input.channel_names
normalization.target.channel_names
normalization.input.mean/std/min/max
normalization.target.mean/std/min/max
normalization.input.units
normalization.target.units
```

The inference service prefers:

- `best_model.pth`

over:

- `model.pth`

## 14. How To Use The Trained Checkpoint In Docker

The default compose setup already points FNO to:

```text
FNO_CHECKPOINT_PATH=/app/artifacts/checkpoints/baseline
FNO_DATASET_PATH=/app/artifacts/datasets/limestone_fno
```

So after training, restart the FNO service:

```powershell
docker compose up -d fno-service backend frontend
```

Then check:

```powershell
curl http://localhost:9002/ready
curl http://localhost:8000/api/v1/models
```

If the dataset and checkpoint are present, `fno-service` should move from fallback mode to checkpoint mode.

For a healthier modern checkpoint, `http://localhost:9002/ready` should show:

- `mode = checkpoint`
- `checkpoint_loaded = true`
- `model_version = fno-baseline@best_model.pth`

If you still see warnings like:

```text
missing_input_normalization_metadata
missing_output_denormalization_metadata
```

then you are still using an old FNO checkpoint and should retrain it with the current code.

You can also test one direct FNO prediction after restart:

```powershell
curl -X POST http://localhost:9002/predict `
  -H "Content-Type: application/json" `
  -d "{\"medium\":{\"id\":\"limestone\",\"name\":\"Limestone\",\"category\":\"sedimentary\",\"properties\":{\"rho\":2694.0,\"vp\":5.8,\"vs\":3.1,\"thermal_conductivity\":2.3,\"heat_capacity\":900.0,\"thermal_expansion\":0.00001}},\"scenario\":{\"temperature_c\":120.0,\"pressure_mpa\":35.0,\"time_ms\":12.0},\"source\":{\"type\":\"thermal_pulse\",\"x\":0.15,\"y\":0.4,\"z\":0.0,\"amplitude\":1.0,\"frequency_hz\":50.0,\"direction\":[1.0,0.0,0.0]},\"probe\":{\"x\":0.7,\"y\":0.55,\"z\":0.0},\"domain\":{\"type\":\"rect_2d\",\"size\":{\"lx\":1.0,\"ly\":1.0,\"lz\":0.0},\"resolution\":{\"nx\":128,\"ny\":128,\"nz\":1},\"boundary_conditions\":{\"left\":\"fixed\",\"right\":\"free\",\"top\":\"insulated\",\"bottom\":\"insulated\"}},\"representation\":\"grid\",\"routing_hint\":\"fno\"}"
```

In a good post-update checkpoint response, watch these fields:

- `diagnostics.normalization_used = true`
- `diagnostics.denormalization_used = true`
- `diagnostics.effective_domain_type = rect_2d`
- `diagnostics.domain_adaptation = none`
- `diagnostics.warnings`

If `diagnostics.warnings` contains only an empty list or small non-scale notes, that is a much better sign than the old huge-output baseline.

If you still see:

```text
scale_outlier:max_displacement_gt_1e2
scale_outlier:max_temperature_perturbation_gt_1e4
```

then the new checkpoint still needs further tuning or a better dataset.

## 15. What Changed In The FNO Stack

The current FNO code now does the following:

- normalizes input channels channel-by-channel during training;
- normalizes target channels channel-by-channel during training;
- stores normalization metadata in the checkpoint;
- denormalizes output channels before computing physical metrics during inference;
- computes `max_displacement` from `sqrt(disp_x^2 + disp_y^2)` for `rect_2d`;
- computes `max_temperature_perturbation` from denormalized temperature relative to `reference_temperature_k`;
- adds explicit warnings for suspicious output scales.

This means:

- **newly trained checkpoints are the ones you should use**;
- old baseline checkpoints may still run, but they are not the right reference for final thesis comparison.

## 16. Current FNO Limitations

Before spending a lot of time on long runs, keep these in mind:

- this is currently a 2D `FNO2d` baseline;
- it expects `Z=1`;
- it is not yet the same maturity level as the full PINN multi-rock experiment stack;
- it does not enforce thermoelastic PDE residuals like the PINN service;
- it is suitable for MVP experimentation and integration, not final scientific claims.

## 17. Recommended Windows Workflow

If you want the safest practical order on Windows, use this:

1. Build or choose one real `structured_dataset.npz`.
2. Convert it with `prepare_fno_dataset.py` using `--grid-res 1 32 32`.
3. Run the 1-epoch smoke training.
4. Run a 20-50 epoch real training command on CUDA.
5. Restart `fno-service`.
6. Check `/ready`.
7. Check one direct `/predict` response and inspect `diagnostics.warnings`.
8. Only after that rerun the full comparison pipeline.
