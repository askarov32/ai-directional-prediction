# PINN Model Structure

This document describes the concrete neural network structure implemented in
`pinn-service`. It is scoped to the actual code in this repository and should
be read as a research prototype architecture note, not as a claim of a
field-validated thermoelastic simulation.

## Source Files

- `src/pinn_service/model.py`: neural network modules.
- `src/pinn_service/training_config.py`: architecture and loss configuration.
- `src/pinn_service/trainer.py`: model creation, optimizer, checkpoint writing.
- `src/pinn_service/inference_service.py`: checkpoint loading and inference.
- `src/pinn_service/losses.py`: hybrid supervised and physics-informed loss.
- `src/pinn_service/physics.py`: thermoelastic residual utilities.

## Public Neural Contract

All PINN variants use the same input and output shape.

Input vector:

```text
X = [x, y, z, t, E, nu, rho, alpha, k, Cp]
```

Concrete feature order:

```text
0: x
1: y
2: z
3: t
4: youngs_modulus
5: poissons_ratio
6: density
7: thermal_expansion
8: thermal_conductivity
9: heat_capacity
```

Output vector:

```text
Y_hat = [T_hat, u_hat, v_hat, w_hat]
```

Concrete output order:

```text
0: temperature_k
1: disp_x
2: disp_y
3: disp_z
```

The training data is scaled before the forward pass. Inference loads the
same `input_scaler` and `output_scaler` from the checkpoint.

## Architecture Variants

The factory function is:

```python
create_pinn_model(
    input_dim=10,
    output_dim=4,
    architecture="mlp" | "res_split",
    hidden_dim=192,
    depth=6,
    activation="tanh",
)
```

Supported activations:

```text
tanh, silu, gelu, relu
```

## Variant 1: Baseline MLP_PINN

Class:

```text
MLP_PINN
```

Default configuration:

```text
architecture = mlp
input_dim = 10
output_dim = 4
hidden_dim = 192
depth = 6
activation = tanh
```

Layer structure:

```text
[B, 10]
  -> Linear(10, 192)
  -> Tanh
  -> Linear(192, 192)
  -> Tanh
  -> Linear(192, 192)
  -> Tanh
  -> Linear(192, 192)
  -> Tanh
  -> Linear(192, 192)
  -> Tanh
  -> Linear(192, 192)
  -> Tanh
  -> Linear(192, 4)
[B, 4]
```

Compact diagram:

```text
Input [x,y,z,t,E,nu,rho,alpha,k,Cp]
        |
        v
Fully connected MLP trunk
6 hidden layers, hidden_dim=192, tanh by default
        |
        v
Output [T,u,v,w]
```

The MLP can also use explicit hidden widths:

```text
--mlp-layer-dims 256,256,192,192,128,128
```

In that mode, `depth` and `hidden_dim` no longer determine the hidden stack.

## Variant 2: ResSplitPINN

Class:

```text
ResSplitPINN
```

Default configuration:

```text
architecture = res_split
input_dim = 10
output_dim = 4
coord_dim = 4
material_dim = 6
hidden_dim = 192
num_blocks = 4
activation = tanh
use_fourier_features = false
```

The model splits inputs into coordinate and material branches.

Coordinate branch:

```text
coordinates = [x, y, z, t]

if Fourier features are disabled:
  [B, 4] -> Linear(4, coord_hidden_dim) -> activation

if Fourier features are enabled:
  [B, 4] -> FourierCoordinateEncoding -> Linear(encoded_dim, coord_hidden_dim) -> activation
```

Material branch:

```text
material = [E, nu, rho, alpha, k, Cp]

[B, 6]
  -> Linear(6, 64)
  -> activation
  -> Linear(64, 64)
  -> activation
```

Fusion and residual trunk:

```text
concat([coordinate_features, material_features])
  -> Linear(coord_hidden_dim + 64, hidden_dim)
  -> activation
  -> ResidualMLPBlock repeated num_blocks times
```

Each residual block:

```text
x
  -> Linear(hidden_dim, hidden_dim)
  -> activation
  -> Linear(hidden_dim, hidden_dim)
  -> add skip connection
  -> activation
```

Split output heads:

```text
temperature_head:
  hidden_dim -> head_hidden_dim -> 1

displacement_head:
  hidden_dim -> head_hidden_dim -> 3
```

Compact diagram:

```text
Coordinates [x,y,z,t] -----------------> coordinate encoder ----\
                                                                  \
                                                                   -> fusion -> residual trunk -> temperature head -> T
                                                                  /                               -> displacement head -> [u,v,w]
Material [E,nu,rho,alpha,k,Cp] -> material encoder ---------------/
```

For the default `hidden_dim=192`:

```text
coord_hidden_dim = max(hidden_dim / 2, 64) = 96
material_hidden_dim = 64
head_hidden_dim = max(hidden_dim / 2, 64) = 96
```

For a faster training configuration such as `hidden_dim=128`:

```text
coord_hidden_dim = 64
material_hidden_dim = 64
head_hidden_dim = 64
```

## Optional Fourier Coordinate Encoding

Class:

```text
FourierCoordinateEncoding
```

This is available only in `ResSplitPINN`.

It expands coordinates as:

```text
[x, y, z, t]
  -> [coords, sin(2*pi*f_i*coords), cos(2*pi*f_i*coords)]
```

Default Fourier settings:

```text
fourier_num_frequencies = 6
fourier_scale = 1.0
```

Encoded coordinate dimension:

```text
coord_dim * (1 + 2 * fourier_num_frequencies)
```

With `coord_dim=4` and `fourier_num_frequencies=6`:

```text
4 * (1 + 12) = 52
```

## Training Loss Structure

Both architectures use the same training objective:

```text
total_loss =
  supervised_weight * supervised_loss
  + velocity_weight * velocity_consistency_loss
  + wave_residual_weight * wave_residual_loss
  + thermal_residual_weight * thermal_residual_loss
```

Default weights:

```text
supervised_weight = 1.0
velocity_weight = 0.25
wave_residual_weight = 0.1
thermal_residual_weight = 0.05
```

Supported physics modes:

```text
coupled_thermoelastic
simple_heat
plane_strain_2d
```

For strict 2D retraining, use:

```text
physics_mode = plane_strain_2d
```

## Optimizer And Checkpoint Contents

Training uses:

```text
optimizer = AdamW
learning_rate = 1e-3 by default
weight_decay = 1e-6 by default
gradient clipping = max_grad_norm=1.0 by default
optional ReduceLROnPlateau scheduler
optional early stopping
```

Checkpoint files store:

```text
model_state_dict
config
input_feature_names
output_feature_names
input_scaler
output_scaler
best_loss
```

During inference, `PINNInferenceService` recreates the exact architecture from
the checkpoint config and then loads `model_state_dict`.

## Recommended Practical Configurations

Small baseline:

```bash
PYTHONPATH=pinn-service/src python -m pinn_service.train \
  --dataset pinn-service/artifacts/rod_experiments_2d/splits/train_samples.npz \
  --val-dataset pinn-service/artifacts/rod_experiments_2d/splits/val_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/baseline_2d \
  --architecture mlp \
  --hidden-dim 192 \
  --depth 6 \
  --physics-mode plane_strain_2d \
  --device cuda
```

Faster structured baseline:

```bash
PYTHONPATH=pinn-service/src python -m pinn_service.train \
  --dataset pinn-service/artifacts/rod_experiments_2d/splits/train_samples.npz \
  --val-dataset pinn-service/artifacts/rod_experiments_2d/splits/val_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/res_split_2d \
  --architecture res_split \
  --hidden-dim 128 \
  --num-blocks 3 \
  --physics-mode plane_strain_2d \
  --batch-size 4096 \
  --validation-batch-size 4096 \
  --device cuda
```

More expressive structured baseline:

```bash
PYTHONPATH=pinn-service/src python -m pinn_service.train \
  --dataset pinn-service/artifacts/rod_experiments_2d/splits/train_samples.npz \
  --val-dataset pinn-service/artifacts/rod_experiments_2d/splits/val_samples.npz \
  --output-dir pinn-service/artifacts/checkpoints/res_split_fourier_2d \
  --architecture res_split \
  --hidden-dim 192 \
  --num-blocks 4 \
  --use-fourier-features \
  --fourier-num-frequencies 6 \
  --physics-mode plane_strain_2d \
  --device cuda
```

## One-Sentence Summary

The current PINN is a hybrid supervised plus physics-informed neural baseline:
the original variant is a plain MLP, while the improved variant separates
coordinate and material features, fuses them through a residual trunk, and
uses separate heads for temperature and displacement.
