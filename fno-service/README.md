# FNO Service

This is the service skeleton for the Fourier Neural Operator route.

Current phase:

```text
Phase 2: service skeleton only
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

Not implemented yet:

- FNO spectral layers;
- training loop;
- checkpoint loading;
- field inference.

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

If `FNO_ALLOW_FALLBACK=true`, `/ready` can report ready and `/predict` returns a deterministic placeholder response. This is only for local stack wiring while the real FNO model is not implemented yet.

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
