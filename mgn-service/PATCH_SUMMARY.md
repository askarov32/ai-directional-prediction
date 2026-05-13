# Patch summary: universal dataset + MeshGraphNet project

This version keeps MeshGraphNet fully working while adding a universal dataset
formatter for future FNO, PINN and Transformer/neural-operator experiments.

## Added

- `scripts/reformat_dataset.py`
  - Reads real COMSOL CSV files and optional `.mphtxt` mesh.
  - Writes model-agnostic `processed/canonical/`.
  - Writes MeshGraphNet adapter `processed/graph/`.
  - Copies `graph.pt` and `trajectories.pt` to `processed/` root for the existing MGN training scripts.
  - Optionally writes FNO regular grid in `processed/fno/`.
  - Writes PINN and Transformer adapter indexes without duplicating large tensors.

- `src/data/universal_formatter.py`
  - Separates dynamic prediction fields from static/material/scenario fields.
  - Builds boundary/source/impact/pressure/building-load masks.
  - Supports kNN fallback when `.mphtxt` is missing or mismatched.
  - Uses existing robust COMSOL parsing: only `data_displacement*`, `data_temperature*`, `data_strain*`, `data_stress*`, `data_materials*` are read as physics CSVs.

- `docs/UNIVERSAL_DATASET.md`
  - Documents canonical, graph, FNO, PINN and Transformer layouts.

## Verified

The formatter was executed on `sandstone_comsol_real`:

```bash
python scripts/reformat_dataset.py --config configs/smoke_test.yaml --dataset_id sandstone_comsol_real --formats canonical graph pinn transformer --k_nearest 8
```

Then MeshGraphNet smoke training was executed:

```bash
python scripts/train_base_model.py --config configs/smoke_test.yaml --dataset_ids sandstone_comsol_real
```

Observed dimensions:

```text
nodes=4448
edges=41426
timesteps=101
dynamic_fields=20
static_features=60
node_in_dim=80
edge_in_dim=4
out_dim=20
```

The mesh had a node-count mismatch with CSV (`33733` vs `4448`), so the project
correctly used kNN fallback instead of failing.

## Important

FNO grids can be very large. Use `--fno_max_timesteps 128` for first experiments
and only use `--fno_max_timesteps all` when you have enough RAM and disk space.
