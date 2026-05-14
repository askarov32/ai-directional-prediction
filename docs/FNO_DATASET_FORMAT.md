# FNO Dataset Format

## Current Directory Layout

`fno-service` expects a regular-grid dataset directory with:

```text
grid_dynamic.npy
grid_static.npy
grid_masks.npy
grid_coords.npy
field_names.json
static_feature_names.json
mask_names.json
metadata.json
```

Optional files:

```text
selected_time_indices.npy
source_node_index.npy
```

## Array Shapes

Current tensor conventions:

```text
grid_dynamic.npy   [T, C, Z, Y, X]
grid_static.npy    [S, Z, Y, X]
grid_masks.npy     [M, Z, Y, X]
grid_coords.npy    [3, Z, Y, X]
```

Where:

- `T`: number of timesteps;
- `C`: dynamic field channels;
- `S`: static/material channels;
- `M`: mask channels.

## Current MVP Assumption

The current `FNO2d` baseline expects:

```text
Z = 1
```

This lets the service squeeze the grid to `[batch, channels, height, width]` for `FNO2d`.

## Channel Semantics

Typical dynamic channels:

```text
temperature_k
disp_x
disp_y
disp_z
```

Typical static channels may include:

```text
youngs_modulus
poissons_ratio
density
thermal_expansion
thermal_conductivity
thermal_density
heat_capacity
```

The actual names are stored in:

- `field_names.json`
- `static_feature_names.json`
- `mask_names.json`

## Model Input Construction

At runtime/training, the current sample builder concatenates:

1. dynamic fields at time `t`;
2. static features;
3. masks;
4. coordinates;
5. normalized time channel.

The target is the selected primary dynamic channels at time `t + 1`.

## Metadata

`metadata.json` should include timing and reference information when available, for example:

```json
{
  "format": "fno_grid",
  "time_start": 0.0,
  "time_end": 30.0,
  "reference_temperature_k": 293.15
}
```

This metadata is used by the current inference layer to align request time and temperature interpretation with the dataset/checkpoint artifacts.
