# Windows MGN Training

This guide is a copy-paste oriented setup for training `mgn-service` on Windows in the new strict 2D workflow.

It assumes:

- you are in the project root;
- you use PowerShell;
- you already built `pinn-service/artifacts/rod_experiments_2d`;
- you want reusable dataset ids inside `mgn-service/datasets`;
- you prefer CUDA when available, but CPU still works.

## 1. Activate the environment

```powershell
.\.venv-pinn\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-pinn\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r mgn-service/requirements.txt
```

## 3. Optional CUDA check

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, you can use CUDA training. If it prints `False`, use `cpu`.

## 4. Build strict 2D MGN datasets

This step converts:

```text
pinn-service/artifacts/rod_experiments_2d/<rock>/structured_dataset.npz
```

into ready-to-train MeshGraphNet dataset ids inside:

```text
mgn-service/datasets/<rock>_rod_2d
```

Command:

```powershell
python mgn-service/scripts/build_2d_mgn_datasets.py `
  --input-root pinn-service/artifacts/rod_experiments_2d `
  --registry-root mgn-service/datasets `
  --k-nearest 12 `
  --target-mode delta
```

This creates, for example:

```text
mgn-service/datasets/granite_rod_2d
mgn-service/datasets/limestone_rod_2d
mgn-service/datasets/sandstone_rod_2d
mgn-service/datasets/basalt_rod_2d
```

and also:

```text
mgn-service/datasets/manifest_2d.json
```

Each dataset contains:

```text
scenario.yaml
processed/graph.pt
processed/trajectories.pt
processed/metadata.json
processed/normalization.json
processed/dynamic_normalization.json
processed/static_normalization.json
processed/preview.csv
```

## 5. Smoke training

Run from the `mgn-service` directory:

```powershell
Push-Location mgn-service
python scripts/train_base_model.py --config configs/train_2d.yaml --dataset_ids limestone_rod_2d --epochs 1
Pop-Location
```

## 6. Main strict 2D baseline training

Single rock example:

```powershell
Push-Location mgn-service
python scripts/train_base_model.py --config configs/train_2d.yaml --dataset_ids limestone_rod_2d --epochs 200
Pop-Location
```

Multi-rock example:

```powershell
Push-Location mgn-service
python scripts/train_base_model.py --config configs/train_2d.yaml --dataset_ids granite_rod_2d limestone_rod_2d sandstone_rod_2d basalt_rod_2d --epochs 200
Pop-Location
```

The 2D config writes to:

```text
mgn-service/outputs/checkpoints_2d
mgn-service/outputs/logs_2d
```

## 7. Connect the trained checkpoint to the service

After training, use `.env` like this:

```env
MGN_DATASET_ID=limestone_rod_2d
MGN_CHECKPOINT_PATH=outputs/checkpoints_2d/best_model.pt
MGN_DEVICE=cpu
```

If Docker GPU is actually configured and working, you may use:

```env
MGN_DEVICE=cuda
```

Then rebuild/restart:

```powershell
docker compose up -d --build mgn-service
```

## 8. Verify readiness

```powershell
curl http://localhost:9001/ready
```

Expected signs:

- `ready: true`
- `mode: rollout`
- `checkpoint_exists: true`
- `dataset_exists: true`

## 9. Recommended workflow

1. Build 2D MGN datasets.
2. Run a 1-epoch smoke training.
3. Run the real 2D baseline training.
4. Point `.env` to `outputs/checkpoints_2d/best_model.pt`.
5. Restart `mgn-service`.
6. Check `/ready`.
7. Re-run the strict 2D comparison pipeline.
