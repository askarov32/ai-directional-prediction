# Universal dataset format for MeshGraphNet, FNO, PINN and Transformer operators

The project keeps the original COMSOL exports in `raw/` and creates a universal
processed layout under `processed/`.

```text
datasets/<dataset_id>/
├── raw/                         # untouched COMSOL CSV + optional MPHTXT
├── scenario.yaml                 # scenario/material/time metadata
└── processed/
    ├── canonical/                # model-agnostic source of truth
    ├── graph/                    # MeshGraphNet adapter
    ├── fno/                      # optional regular grid adapter
    ├── pinn/                     # sampling index, no duplicated tensors
    ├── transformer/              # token sampling index, no duplicated tensors
    ├── graph.pt                  # legacy MeshGraphNet copy
    ├── trajectories.pt           # legacy MeshGraphNet copy
    ├── metadata.json
    └── normalization.json
```

## 1. Canonical source of truth

`processed/canonical/` contains full physical data on the COMSOL node set:

```text
coords.npy                  [N, 3]
time.npy                    [T]
dynamic.npy                 [T, N, F_dyn]     raw physical units
dynamic_norm.npy            [T, N, F_dyn]     normalized
static.npy                  [N, F_static]     raw static conditioning
static_norm.npy             [N, F_static]     normalized
masks.npy                   [N, F_mask]
field_names.json
static_feature_names.json
mask_names.json
metadata.json
normalization.json
```

Dynamic fields are the prediction target, for example `T`, `u`, `v`, `w`,
`ut`, `vt`, `wt`, stress and strain components. Material parameters are not
predicted. They are moved into `static.npy` / `static_norm.npy`.

## 2. MeshGraphNet adapter

MeshGraphNet uses:

```text
processed/graph/graph.pt
processed/graph/trajectories.pt
```

The same files are also copied to `processed/graph.pt` and
`processed/trajectories.pt` so the existing training scripts keep working.

`graph.pt` contains:

```python
{
  "coords": Tensor[N, 3],
  "edge_index": Tensor[2, E],
  "edge_attr": Tensor[E, 4],        # dx, dy, dz, distance
  "node_static": Tensor[N, S],
  "node_static_raw": Tensor[N, S],
  "static_feature_names": list[str],
  "field_names": list[str]
}
```

`trajectories.pt` contains train/val/test pairs:

```python
x_t = concat(node_static_norm, state_norm[t])
y   = state_norm[t+1] - state_norm[t]     # target_mode = delta
```

## 3. FNO adapter

FNO requires a regular grid. The script can export:

```text
processed/fno/grid_dynamic.npy      [T_selected, C_dyn, Z, Y, X]
processed/fno/grid_static.npy       [C_static, Z, Y, X]
processed/fno/grid_masks.npy        [C_mask, Z, Y, X]
processed/fno/grid_coords.npy       [3, Z, Y, X]
processed/fno/source_node_index.npy [Z*Y*X]
```

The current implementation uses nearest-node projection from COMSOL nodes to the
regular grid. For first experiments this is robust and has no SciPy dependency.
For high-quality FNO experiments you can later replace it with linear/RBF
interpolation while keeping the same file contract.

Full 3D FNO grids can become huge. Use `--fno_max_timesteps` to export a subset
first.

## 4. PINN and Transformer adapters

`processed/pinn/index.json` and `processed/transformer/index.json` do not copy
large tensors. They point to canonical arrays and describe how loaders should
sample:

- PINN: sample `(x, y, z, t)` collocation/data points from `canonical/`;
- Transformer/neural operator: sample input tokens and query tokens from
  canonical nodes and time indices.

## 5. Main command

```bash
python scripts/reformat_dataset.py \
  --config configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats canonical graph pinn transformer \
  --k_nearest 12
```

With FNO grid:

```bash
python scripts/reformat_dataset.py \
  --config configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats canonical graph fno pinn transformer \
  --grid_res 32 32 32 \
  --fno_max_timesteps 128
```

Full FNO sequence, only when you have enough disk/RAM:

```bash
python scripts/reformat_dataset.py \
  --config configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats fno \
  --grid_res 32 32 32 \
  --fno_max_timesteps all
```

## 6. Recommended workflow

For your current MeshGraphNet work:

```bash
python scripts/reformat_dataset.py --dataset_id sandstone_comsol_real --formats canonical graph pinn transformer
python scripts/train_base_model.py --config configs/base.yaml --dataset_ids sandstone_comsol_real
```

For future FNO experiments:

```bash
python scripts/reformat_dataset.py --dataset_id sandstone_comsol_real --formats fno --grid_res 32 32 32 --fno_max_timesteps 128
```
